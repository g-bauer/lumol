// Cymbalum, an extensible molecular simulation engine
// Copyright (C) 2015-2016 G. Fraux — BSD license

//! Ewald summation of coulombic potential
use special::{erfc, erf};
use std::f64::consts::{PI, FRAC_2_SQRT_PI};
use std::f64;
use std::cell::{Cell, RefCell};

use system::{System, UnitCell, CellType};
use types::{Matrix3, Vector3D, Array3, Complex, Zero};
use constants::ELCC;
use potentials::PairRestriction;

use super::{GlobalPotential, CoulombicPotential, DefaultGlobalCache};

/// Ewald summation of the coulombic interactions. The Ewald summation is based
/// on a separation of the coulombic potential U in two parts, using the trivial
/// identity:
///
///     U(x) = U(x) * (f(x) + 1) - U(x) * f(x)
///
/// where `f(x)` is the `erf` function. This leads to a separation of the
/// conditionally convergent coulombic sum into two absolutly convergent sums:
/// one in real space, and the other in Fourier or k-space.
///
/// For more informations about this algorithm see [FS2002].
///
/// [FS2002] Frenkel, D. & Smit, B. Understanding molecular simulation. (Academic press, 2002).
#[derive(Clone, Debug)]
pub struct Ewald {
    /// Splitting parameter between k-space and real space
    alpha: f64,
    /// Cutoff radius in real space
    rc: f64,
    /// Number of points to use in k-space
    kmax: usize,
    /// Cutoff in k-space
    kmax2: Cell<f64>,
    /// Restriction scheme
    restriction: PairRestriction,
    /// Caching exponential factors `exp(-k^2 / (4 alpha^2)) / k^2`.
    ///
    /// k vectors are indexed by three isize number, in the intervals
    /// `[0, kmax] x [-kmax, kmax] x [-kmax, kmax]`. Because we do not have
    /// negative integers indexing, we use `[0, kmax] x [0, 2kmax+1] x [0, 2kmax+1]`,
    /// and translate all the indexes.
    expfactors: RefCell<Array3<f64>>,
    /// Phases for the Fourier transform.
    ///
    /// This is indexed by `[0, 2*kmax+1] x [0, natoms - 1] x [0, 2]`, with the
    /// same translation in the first index that for `expfactors`.
    fourier_factors: RefCell<Array3<Complex>>,
    /// Fourier transform of the electrostatic density
    rho: RefCell<Array3<Complex>>,
    /// Guard for cache invalidation of expfactors
    previous_cell: Cell<Option<UnitCell>>,
}

impl Ewald {
    /// Create an Ewald summation using the `rc` cutoff radius in real space,
    /// and `kmax` points in k-space (Fourier space).
    pub fn new(rc: f64, kmax: usize) -> Ewald {
        let ksize = 2 * kmax + 1;
        let expfactors = Array3::zeros((kmax + 1, ksize, ksize));
        let rho = Array3::zeros((kmax + 1, ksize, ksize));
        let alpha = 3.0 * PI / (4.0 * rc);
        Ewald {
            alpha: alpha,
            rc: rc,
            kmax: kmax,
            kmax2: Cell::new(0.0),
            restriction: PairRestriction::None,
            expfactors: RefCell::new(expfactors),
            fourier_factors: RefCell::new(Array3::zeros((0, 0, 0))),
            rho: RefCell::new(rho),
            previous_cell: Cell::new(None),
        }
    }
}

/// Real space part of the summation
impl Ewald {
    /// Real space contribution to the energy
    fn real_space_energy(&self, system: &System) -> f64 {
        let natoms = system.size();
        let mut energy = 0.0;
        for i in 0..natoms {
            for j in (i + 1)..natoms {
                if self.restriction.is_excluded_pair(system, i, j) {continue}
                let s = self.restriction.scaling(system, i, j);
                assert!(s == 1.0, "Scaling restriction scheme using Ewald are not implemented");

                let r = system.distance(i, j);
                if r > self.rc {continue};

                energy += s * system[i].charge * system[j].charge * erfc(self.alpha * r) / r;
            }
        }
        return energy / ELCC;
    }

    /// Real space contribution to the forces
    fn real_space_forces(&self, system: &System, res: &mut Vec<Vector3D>) {
        let natoms = system.size();
        assert!(res.len() == system.size());

        for i in 0..natoms {
            for j in (i + 1)..natoms {
                if self.restriction.is_excluded_pair(system, i, j) {continue}
                let s = self.restriction.scaling(system, i, j);
                assert!(s == 1.0, "Scaling restriction scheme using Ewald are not implemented");

                let rij = system.wraped_vector(i, j);
                let r = rij.norm();
                if r > self.rc {continue};

                let factor = s * self.real_space_force_factor(r, system[i].charge, system[j].charge);
                let force = factor * rij;
                res[i] = res[i] + force;
                res[j] = res[j] - force;
            }
        }
    }

    /// Get the real-space force factor at distance `r` for charges `qi` and `qj`
    #[inline]
    fn real_space_force_factor(&self, r: f64, qi: f64, qj: f64) -> f64 {
        let mut factor = erfc(self.alpha * r) / r;
        factor += self.alpha * FRAC_2_SQRT_PI * f64::exp(-self.alpha * self.alpha * r * r);
        factor *= qi * qj / (r * r) / ELCC;
        return factor;
    }

    /// Real space contribution to the virial
    fn real_space_virial(&self, system: &System) -> Matrix3 {
        let natoms = system.size();
        let mut res = Matrix3::zero();
        for i in 0..natoms {
            for j in (i + 1)..natoms {
                if self.restriction.is_excluded_pair(system, i, j) {continue}
                let s = self.restriction.scaling(system, i, j);
                assert!(s == 1.0, "Scaling restriction scheme using Ewald are not implemented");

                let rij = system.wraped_vector(i, j);
                let r = rij.norm();
                if r > self.rc {continue};

                let factor = s * self.real_space_force_factor(r, system[i].charge, system[j].charge);
                let force = -factor * rij;

                res = res + force.tensorial(&rij);
            }
        }
        return res;
    }
}

/// Self-interaction corection
impl Ewald {
    /// Self-interaction contribution to the energy
    fn self_energy(&self, system: &System) -> f64 {
        let mut q2 = 0.0;
        for i in 0..system.size() {
            q2 += system[i].charge * system[i].charge;
        }
        return -self.alpha / f64::sqrt(PI) * q2 / ELCC;
    }
}

/// k-space part of the summation
impl Ewald {
    /// Get the index in any array indexed by k point from the 3 k indexes. This
    /// function assumes that the indexes are in the [-kmax, kmax] range.
    #[inline]
    fn get_idx(&self, kx: isize, ky: isize, kz: isize) -> (usize, usize, usize) {
        let ky = (ky + self.kmax as isize) as usize;
        let kz = (kz + self.kmax as isize) as usize;
        return (kx as usize, ky, kz)
    }

    /// Translate a k index from [-kmax, kmax] to [0, 2kmax + 1].
    #[inline]
    fn trans_idx(&self, k: isize) -> usize {
        (k + self.kmax as isize) as usize
    }

    fn precompute(&self, cell: &UnitCell) {
        if let Some(ref prev_cell) = self.previous_cell.get() {
            if cell == prev_cell {
                // Do not recompute
                return;
            }
        }

         match cell.celltype() {
            CellType::Infinite => {
                error!("Can not use Ewald sum with Infinite cell.");
                panic!();
            },
            CellType::Triclinic | CellType::Orthorombic => {
                // All good!
            },
        }

        self.previous_cell.set(Some(*cell));
        let mut expfactors = self.expfactors.borrow_mut();

        // Because we do a spherical truncation in k space, we have to transform
        // kmax into a spherical cutoff 'radius'
        let lenghts = cell.lengths();
        let max_lenght = f64::max(f64::max(lenghts.0, lenghts.1), lenghts.2);
        let min_lenght = f64::min(f64::min(lenghts.0, lenghts.1), lenghts.2);
        let k_rc = self.kmax as f64 * (2.0 * PI / max_lenght);
        self.kmax2.set(k_rc * k_rc);

        if self.rc > min_lenght / 2.0 {
            warn!("The Ewald cutoff is too high for this unit cell, energy might be wrong.");
        }

        // Now, we precompute the exp(-k^2/4a^2)/k^2 terms. We use the symmetry
        // to only store ikx >= 0 terms
        let ikmax = self.kmax as isize;
        let (rec_vx, rec_vy, rec_vz) = cell.reciprocal_vectors();
        for ikx in 0..(ikmax + 1) {
            for iky in (-ikmax)..(ikmax + 1) {
                for ikz in (-ikmax)..(ikmax + 1) {
                    let k = (ikx as f64) * rec_vx + (iky as f64) * rec_vy + (ikz as f64) * rec_vz;
                    let k2 = k.norm2();
                    let idx = self.get_idx(ikx, iky, ikz);
                    if k2 > self.kmax2.get() {
                        expfactors[idx] = 0.0;
                        continue;
                    }
                    expfactors[idx] = f64::exp(-k2 / (4.0 * self.alpha * self.alpha)) / k2;
                    if ikx != 0 {expfactors[idx] *= 2.0;}
                }
            }
        }
        let idx = self.get_idx(0, 0, 0);
        expfactors[idx] = 0.0;
    }

    /// Compute the Fourier transform of the electrostatic density
    fn density_fft(&self, system: &System) {
        let mut fourier_factors = self.fourier_factors.borrow_mut();

        let ksize = 2 * self.kmax + 1;
        let natoms = system.size();
        fourier_factors.resize_if_different((ksize, natoms, 3));

        // Do the k=(-1, 0, 1) cases first
        for i in 0..natoms {
            let ri = system.cell().fractional(&system[i].position);
            for j in 0..3 {
                fourier_factors[(self.trans_idx(0), i, j)] = Complex::cartesian(1.0, 0.0);
                fourier_factors[(self.trans_idx(1), i, j)] = Complex::polar(1.0, -2.0 * PI * ri[j]);
                fourier_factors[(self.trans_idx(-1), i, j)] = fourier_factors[(self.trans_idx(1), i, j)].conj();
            }
        }

        let ikmax = self.kmax as isize;
        // Use recursive definition for computing the factor for all the other values of k.
        for ik in 2..(ikmax + 1) {
            for i in 0..natoms {
                for j in 0..3 {
                    fourier_factors[(self.trans_idx(ik), i, j)] = fourier_factors[(self.trans_idx(ik - 1), i, j)] * fourier_factors[(self.trans_idx(1), i, j)];
                    fourier_factors[(self.trans_idx(-ik), i, j)] = fourier_factors[(self.trans_idx(ik), i, j)].conj();
                }
            }
        }

        let mut rho = self.rho.borrow_mut();
        for kx in 0..(self.kmax + 1) {
            for ky in 0..(2*self.kmax + 1) {
                for kz in 0..(2*self.kmax + 1) {
                    rho[(kx, ky, kz)] = Complex::cartesian(0.0, 0.0);
                    for i in 0..natoms {
                        let rho_i = system[i].charge * fourier_factors[(kx, i, 0)] * fourier_factors[(ky, i, 1)] * fourier_factors[(kz, i, 2)];
                        rho[(kx, ky, kz)] = rho[(kx, ky, kz)] + rho_i;
                    }
                }
            }
        }
    }

    /// k-space contribution to the energy
    fn kspace_energy(&self, system: &System) -> f64 {
        self.density_fft(system);
        let mut energy = 0.0;

        let expfactors = self.expfactors.borrow();
        let rho = self.rho.borrow();
        for kx in 0..(self.kmax + 1) {
            for ky in 0..(2*self.kmax + 1) {
                for kz in 0..(2*self.kmax + 1) {
                    // The k = 0 case and the cutoff in k-space are already
                    // handled in expfactors
                    if expfactors[(kx, ky, kz)].abs() < f64::MIN {continue}
                    let rho2 = (rho[(kx, ky, kz)] * rho[(kx, ky, kz)].conj()).real();
                    energy += expfactors[(kx, ky, kz)] * rho2;
                }
            }
        }
        energy *= 2.0 * PI / (system.cell().volume() * ELCC);
        return energy;
    }

    /// k-space contribution to the forces
    fn kspace_forces(&self, system: &System, res: &mut Vec<Vector3D>) {
        assert!(res.len() == system.size());
        self.density_fft(system);

        let factor = 4.0 * PI / (system.cell().volume() * ELCC);
        let (rec_vx, rec_vy, rec_vz) = system.cell().reciprocal_vectors();

        let expfactors = self.expfactors.borrow();
        let ikmax = self.kmax as isize;
        for ikx in 0..(ikmax + 1) {
            for iky in (-ikmax)..(ikmax + 1) {
                for ikz in (-ikmax)..(ikmax + 1) {
                    // The k = 0 and the cutoff in k-space are already handled in
                    // expfactors.
                    let idx = self.get_idx(ikx, iky, ikz);
                    if expfactors[idx].abs() < f64::MIN {continue}
                    let k = (ikx as f64) * rec_vx + (iky as f64) * rec_vy + (ikz as f64) * rec_vz;
                    for i in 0..system.size() {
                        let qi = system[i].charge;
                        for j in (i + 1)..system.size() {
                            let qj = system[j].charge;
                            let force = factor * self.kspace_force_factor(i, j, idx.0, idx.1, idx.2, qi, qj) * k;

                            res[i] = res[i] + force;
                            res[j] = res[j] - force;
                        }
                    }
                }
            }
        }
    }

    /// Get the force factor for particles `i` and `j` with charges `qi` and
    /// `qj`, at k point  `(kx, ky, kz)`. `(kx, ky, kz)` must be the indexes
    /// of the arrays, already translated.
    #[inline(always)]
    fn kspace_force_factor(&self, i: usize, j: usize, kx: usize, ky: usize, kz: usize, qi: f64, qj: f64) -> f64 {
        let fourier_factors = self.fourier_factors.borrow();
        let expfactors = self.expfactors.borrow();

        let fourier_i = fourier_factors[(kx, i, 0)] * fourier_factors[(ky, i, 1)] * fourier_factors[(kz, i, 2)];
        let fourier_j = fourier_factors[(kx, j, 0)] * fourier_factors[(ky, j, 1)] * fourier_factors[(kz, j, 2)];
        let sin_kr = (fourier_i * fourier_j.conj()).imag();

        return qi * qj * expfactors[(kx, ky, kz)] * sin_kr;
    }

    /// k-space contribution to the virial
    fn kspace_virial(&self, system: &System) -> Matrix3 {
        self.density_fft(system);
        let mut res = Matrix3::zero();

        let factor = 4.0 * PI / (system.cell().volume() * ELCC);
        let (rec_vx, rec_vy, rec_vz) = system.cell().reciprocal_vectors();

        let expfactors = self.expfactors.borrow();
        let ikmax = self.kmax as isize;
        for ikx in 0..(ikmax + 1) {
            for iky in (-ikmax)..(ikmax + 1) {
                for ikz in (-ikmax)..(ikmax + 1) {
                    // The k = 0 and the cutoff in k-space are already handled in
                    // expfactors.
                    let idx = self.get_idx(ikx, iky, ikz);
                    if expfactors[idx].abs() < f64::MIN {continue}
                    let k = (ikx as f64) * rec_vx + (iky as f64) * rec_vy + (ikz as f64) * rec_vz;
                    for i in 0..system.size() {
                        let qi = system[i].charge;
                        for j in (i + 1)..system.size() {
                            let qj = system[j].charge;
                            let force = factor * self.kspace_force_factor(i, j, idx.0, idx.1, idx.2, qi, qj) * k;
                            let rij = system.wraped_vector(i, j);

                            res = res + force.tensorial(&rij);
                        }
                    }
                }
            }
        }
        return res;
    }
}

/// Molecular correction for Ewald summation
impl Ewald {
    /// Molecular correction contribution to the energy
    fn molcorrect_energy(&self, system: &System) -> f64 {
        let natoms = system.size();
        let mut energy = 0.0;

        for i in 0..natoms {
            let qi = system[i].charge;
            // I can not manage to get this work with a loop from (i+1) to N. The finite
            // difference test (testing that the force is the same that the finite difference
            // of the energy) always fail. So let's use it that way for now.
            for j in 0..natoms {
                if i == j {continue}
                // Only account for excluded pairs
                if !self.restriction.is_excluded_pair(system, i, j) {continue}
                let s = self.restriction.scaling(system, i, j);

                let qj = system[j].charge;
                let r = system.distance(i, j);
                assert!(r < self.rc, "Atoms in molecule are separated by more than the cutoff radius of Ewald sum.");

                energy += 0.5 * qi * qj * s / ELCC * erf(self.alpha * r)/r;
            }
        }
        return energy;
    }

    /// Molecular correction contribution to the forces
    fn molcorrect_forces(&self, system: &System, res: &mut Vec<Vector3D>) {
        let natoms = system.size();
        assert!(res.len() == natoms);

        for i in 0..natoms {
            let qi = system[i].charge;
            for j in 0..natoms {
                if i == j {continue}
                // Only account for excluded pairs
                if !self.restriction.is_excluded_pair(system, i, j) {continue}
                let s = self.restriction.scaling(system, i, j);

                let qj = system[j].charge;
                let rij = system.wraped_vector(i, j);
                let r = rij.norm();
                assert!(r < self.rc, "Atoms in molecule are separated by more than the cutoff radius of Ewald sum.");

                let factor = s * self.molcorrect_force_factor(qi, qj, r);
                res[i] = res[i] - factor * rij;
            }
        }
    }

    /// Get the force factor for particles with charges `qi` and `qj`, at
    /// distance `r`.
    #[inline]
    fn molcorrect_force_factor(&self, qi: f64, qj: f64, r: f64) -> f64 {
        qi * qj / (ELCC * r * r) * (2.0 * self.alpha / f64::sqrt(PI) * f64::exp(-self.alpha * self.alpha * r * r) - erf(self.alpha * r) / r)
    }

    /// Molecular correction contribution to the virial
    fn molcorrect_virial(&self, system: &System) -> Matrix3 {
        let natoms = system.size();
        let mut res = Matrix3::zero();

        for i in 0..natoms {
            let qi = system[i].charge;
            for j in 0..natoms {
                if i == j {continue}
                // Only account for excluded pairs
                if !self.restriction.is_excluded_pair(system, i, j) {continue}
                let s = self.restriction.scaling(system, i, j);

                let qj = system[j].charge;
                let rij = system.wraped_vector(i, j);
                let r = rij.norm();
                assert!(r < self.rc, "Atoms in molecule are separated by more than the cutoff radius of Ewald sum.");

                let force = s * self.molcorrect_force_factor(qi, qj, r) * rij;
                res = res + force.tensorial(&rij);
            }
        }
        return res;
    }
}

impl GlobalPotential for Ewald {
    fn energy(&self, system: &System) -> f64 {
        self.precompute(system.cell());
        let real = self.real_space_energy(system);
        let self_e = self.self_energy(system);
        let kspace = self.kspace_energy(system);
        let molecular = self.molcorrect_energy(system);
        return real + self_e + kspace + molecular;
    }

    fn forces(&self, system: &System) -> Vec<Vector3D> {
        self.precompute(system.cell());
        let mut res = vec![Vector3D::new(0.0, 0.0, 0.0); system.size()];
        self.real_space_forces(system, &mut res);
        /* No self force */
        self.kspace_forces(system, &mut res);
        self.molcorrect_forces(system, &mut res);
        return res;
    }

    fn virial(&self, system: &System) -> Matrix3 {
        self.precompute(system.cell());
        let real = self.real_space_virial(system);
        /* No self virial */
        let kspace = self.kspace_virial(system);
        let molecular = self.molcorrect_virial(system);
        return real + kspace + molecular;
    }
}

impl CoulombicPotential for Ewald {
    fn set_restriction(&mut self, restriction: PairRestriction) {
        self.restriction = restriction;
    }
}

impl DefaultGlobalCache for Ewald {}

#[cfg(test)]
mod tests {
    pub use super::*;
    use system::{System, UnitCell, Particle};
    use types::Vector3D;
    use potentials::GlobalPotential;

    const E_BRUTE_FORCE: f64 = -0.09262397663346732;

    pub fn testing_system() -> System {
        let mut system = System::from_cell(UnitCell::cubic(20.0));

        system.add_particle(Particle::new("Cl"));
        system[0].charge = -1.0;
        system[0].position = Vector3D::new(0.0, 0.0, 0.0);

        system.add_particle(Particle::new("Na"));
        system[1].charge = 1.0;
        system[1].position = Vector3D::new(1.5, 0.0, 0.0);

        return system;
    }

    #[test]
    #[should_panic]
    fn infinite_cell() {
        let mut system = testing_system();
        system.set_cell(UnitCell::new());
        let ewald = Ewald::new(8.0, 10);
        ewald.energy(&system);
    }

    #[test]
    #[should_panic]
    fn triclinic_cell() {
        let mut system = testing_system();
        system.set_cell(UnitCell::triclinic(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
        let ewald = Ewald::new(8.0, 10);
        ewald.energy(&system);
    }

    #[test]
    fn indexing() {
        let ewald = Ewald::new(10.0, 10);

        assert_eq!(ewald.trans_idx(-10), 0);
        assert_eq!(ewald.trans_idx(0), 10);
        assert_eq!(ewald.trans_idx(10), 20);

        assert_eq!(ewald.get_idx(0, -5, 4), (0, 5, 14));
        assert_eq!(ewald.get_idx(0, 3, -4), (0, 13, 6));

        assert_eq!(ewald.get_idx(0, 10, -10), (0, 20, 0));
    }

    #[test]
    fn energy() {
        let system = testing_system();
        let ewald = Ewald::new(10.0, 10);

        let e = ewald.energy(&system);
        assert_approx_eq!(e, E_BRUTE_FORCE, 1e-5);
    }

    #[test]
    fn forces() {
        let mut system = testing_system();
        let ewald = Ewald::new(10.0, 10);

        // Finite difference by ewald component
        let real = ewald.real_space_energy(&system);
        let kspace = ewald.kspace_energy(&system);

        const EPS: f64 = 1e-9;
        system[0].position.x += EPS;
        let real1 = ewald.real_space_energy(&system);
        let kspace1 = ewald.kspace_energy(&system);

        let mut forces = vec![Vector3D::new(0.0, 0.0, 0.0); 2];
        ewald.real_space_forces(&system, &mut forces);
        let force = forces[0].x;
        assert_approx_eq!((real - real1)/EPS, force, 1e-6);

        let mut forces = vec![Vector3D::new(0.0, 0.0, 0.0); 2];
        ewald.kspace_forces(&system, &mut forces);
        let force = forces[0].x;
        assert_approx_eq!((kspace - kspace1)/EPS, force, 1e-6);

        // Finite difference computation of the total force
        let e = ewald.energy(&system);
        system[0].position[0] += EPS;

        let e1 = ewald.energy(&system);
        let force = ewald.forces(&system)[0][0];
        assert_approx_eq!((e - e1)/EPS, force, 1e-6);
    }

    #[test]
    fn virial() {
        let system = testing_system();
        let ewald = Ewald::new(10.0, 10);

        let virial = ewald.virial(&system);

        let force = ewald.forces(&system)[0];
        let w = force.tensorial(&Vector3D::new(1.5, 0.0, 0.0));

        for i in 0..3 {
            for j in 0..3 {
                assert_approx_eq!(virial[(i, j)], w[(i, j)]);
            }
        }
    }
}
