/* Cymbalum, Molecular Simulation in Rust - Copyright (C) 2015 Guillaume Fraux
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/
 */
//! Global potential are potentials acting on the whole system at once
//!
//! They can be coulombic potentials, or external provided potential function
//! for example.
extern crate special;
use self::special::erfc;

use std::f64::consts::PI;

use universe::Universe;
use types::{Matrix3, Vector3D};
use constants::ELCC;

/// The `GlobalPotential` trait represent a potential acting on the whole
/// universe at once.
pub trait GlobalPotential {
    /// Compute the energetic contribution of this potential
    fn energy(&self, universe: &Universe) -> f64;
    /// Compute the force contribution of this potential
    fn forces(&self, universe: &Universe) -> Vec<Vector3D>;
    /// Compute the virial contribution of this potential
    fn virial(&self, universe: &Universe) -> Matrix3;
}

/******************************************************************************/
/// Wolf summation for the coulombic potential, as defined in [Wolf1999]. This
/// is a fast direct pairwise summation for coulombic potential.
///
/// [Wolf1999]: Wolf, D. et al. J. Chem. Phys. 110, 8254 (1999).
pub struct Wolf {
    /// Spliting parameter
    alpha: f64,
    /// Cutoff radius in real space
    cutoff: f64,
    /// Energy constant for wolf computation
    energy_cst: f64,
    /// Force constant for wolf computation
    force_cst: f64,
}

impl Wolf {
    /// Create a new Wolf summation, using a real-space cutoff of `cutoff`.
    pub fn new(cutoff: f64) -> Wolf {
        assert!(cutoff > 0.0, "Got a negative cutoff in Wolf summation");
        let alpha = PI/cutoff;
        let e_cst = erfc(alpha*cutoff)/cutoff;
        let f_cst = erfc(alpha*cutoff)/(cutoff*cutoff) + 2.0*alpha/f64::sqrt(PI) * f64::exp(-alpha*alpha*cutoff*cutoff)/cutoff;
        Wolf{
            alpha: alpha,
            cutoff: cutoff,
            energy_cst: e_cst,
            force_cst: f_cst,
        }
    }

    /// Compute the energy for the pair of particles with charge `qi` and `qj`,
    /// at the distance of `rij`.
    #[inline]
    fn energy_pair(&self, qi: f64, qj: f64, rij: f64) -> f64 {
        if rij > self.cutoff {
            return 0.0;
        }
        qi * qj * (erfc(self.alpha*rij)/rij - self.energy_cst) / ELCC
    }

    /// Compute the energy for self interaction of a particle with charge `qi`
    #[inline]
    fn energy_self(&self, qi: f64) -> f64 {
        qi * qi * (self.energy_cst/2.0 + self.alpha/f64::sqrt(PI)) / ELCC
    }

    /// Compute the force for self the pair of particles with charge `qi` and
    /// `qj`, at the distance of `rij`.
    #[inline]
    fn force_pair(&self, qi: f64, qj: f64, rij: Vector3D) -> Vector3D {
        let d = rij.norm();
        if d > self.cutoff {
            return Vector3D::new(0.0, 0.0, 0.0);
        }
        let factor = erfc(self.alpha*d)/(d*d) + 2.0*self.alpha/f64::sqrt(PI) * f64::exp(-self.alpha*self.alpha*d*d)/d;
        return qi * qj * (factor - self.force_cst) * rij.normalized() / ELCC;
    }
}

impl GlobalPotential for Wolf {
    fn energy(&self, universe: &Universe) -> f64 {
        let natoms = universe.size();
        let mut res = 0.0;
        for i in 0..natoms {
            let qi = universe[i].charge();
            for j in i+1..natoms {
                let qj = universe[j].charge();
                if qi*qj == 0.0 {
                    continue;
                }
                let rij = universe.distance(i, j);
                res += self.energy_pair(qi, qj, rij);
            }
            // Remove self term
            res -= self.energy_self(qi);
        }
        return res;
    }

    fn forces(&self, universe: &Universe) -> Vec<Vector3D> {
        let natoms = universe.size();
        let mut res = vec![Vector3D::new(0.0, 0.0, 0.0); natoms];
        for i in 0..natoms {
            let qi = universe[i].charge();
            for j in i+1..natoms {
                let qj = universe[j].charge();
                if qi*qj == 0.0 {
                    continue;
                }
                let rij = universe.wrap_vector(i, j);
                let force = self.force_pair(qi, qj, rij);
                res[i] = res[i] - force;
                res[j] = res[j] + force;
            }
        }
        return res;
    }

    fn virial(&self, universe: &Universe) -> Matrix3 {
        let natoms = universe.size();
        let mut res = Matrix3::zero();
        for i in 0..natoms {
            let qi = universe[i].charge();
            for j in i+1..natoms {
                let qj = universe[j].charge();
                if qi*qj == 0.0 {
                    continue;
                }
                let rij = universe.wrap_vector(i, j);
                let force = self.force_pair(qi, qj, rij);
                res = res + force.tensorial(&rij);
            }
        }
        return res;
    }
}

/******************************************************************************/

#[cfg(test)]
mod tests {
    pub use super::*;
    use universe::{Universe, UnitCell, Particle};
    use types::Vector3D;

    const E_BRUTE_FORCE: f64 = -0.09262397663346732;

    pub fn testing_universe() -> Universe {
        let mut universe = Universe::from_cell(UnitCell::cubic(20.0));

        universe.add_particle(Particle::new("Cl"));
        universe[0].set_charge(-1.0);
        universe[0].set_position(Vector3D::new(0.0, 0.0, 0.0));

        universe.add_particle(Particle::new("Na"));
        universe[1].set_charge(1.0);
        universe[1].set_position(Vector3D::new(1.5, 0.0, 0.0));

        return universe;
    }

    #[test]
    fn wolf() {
        let mut universe = testing_universe();
        let wolf = Wolf::new(8.0);

        let e = wolf.energy(&universe);
        // Wolf is not very good for inhomogeneous systems
        assert_approx_eq!(e, E_BRUTE_FORCE, 1e-2);

        let forces = wolf.forces(&universe);
        let norm = (forces[0] + forces[1]).norm();
        // Total force should be null
        assert_approx_eq!(norm, 0.0, 1e-9);

        let eps = 1e-9;
        let mut pos = universe[0].position().clone();
        pos.x += eps;
        universe[0].set_position(pos);

        let e1 = wolf.energy(&universe);
        let force = wolf.forces(&universe)[0].x;
        // Finite difference computation of the force
        assert_approx_eq!((e1 - e)/eps, force, 1e-6);
    }
}