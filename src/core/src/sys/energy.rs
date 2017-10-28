// Lumol, an extensible molecular simulation engine
// Copyright (C) Lumol's contributors — BSD license

//! Primitives for energy computation.
//!
//! This module provides simple function to compute separated components of the
//! potential energy of an `System`.

use std::f64::consts::PI;

use sys::ParticleKind;

use sys::System;
use parallel::prelude::*;

/// An helper struct to evaluate energy components of a system.
pub struct EnergyEvaluator<'a> {
    system: &'a System,
}

impl<'a> EnergyEvaluator<'a> {
    /// Create a new `EnergyEvaluator` acting on the given `system`.
    pub fn new(system: &'a System) -> EnergyEvaluator<'a> {
        EnergyEvaluator { system: system }
    }
}

impl<'a> EnergyEvaluator<'a> {
    /// Compute the energy associated with the pair of particles `i, j` at
    /// distance `r`
    #[inline]
    pub fn pair(&self, r: f64, i: usize, j: usize) -> f64 {
        let distance = self.system.bond_distance(i, j);
        let mut energy = 0.0;
        for potential in self.system.pair_potentials(i, j) {
            let info = potential.restriction().information(distance);
            if !info.excluded {
                energy += info.scaling * potential.energy(r);
            }
        }
        energy
    }

    /// Compute the energy of all the pairs in the system
    pub fn pairs(&self) -> f64 {

        (0..self.system.size())
            .par_map(|i| {
                let mut local_energy = 0.0;

                for j in (i + 1)..self.system.size() {
                    let r = self.system.nearest_image(i, j).norm();
                    local_energy += self.pair(r, i, j);
                }
                local_energy
            })
            .sum()
    }

    fn pairs_same_kinds(&self, kind: ParticleKind) -> f64 {
        let kind_indices = self.system.particles().kind.iter().enumerate();
        let ids_i: Vec<usize> =
            kind_indices.filter(|&(_, &k)| k == kind).map(|(id, _)| id).collect();
        let ids_j = ids_i.clone();

        if ids_i.len() == 0 || ids_j.len() == 0 {
            return 0.0;
        };
        ids_i.iter()
            .enumerate()
            .map(|(id, &i)| {
                let mut local_energy = 0.0;
                for &j in ids_j.iter().skip(id + 1) {
                    let r = self.system.nearest_image(i, j).norm();
                    local_energy += self.pair(r, i, j);
                    println!("i: {}, j: {}, ener: {}", i, j, local_energy)
                }
                local_energy
            })
            .sum()
    }

    /// Compute energy for all pairs of particles of kind `i` and `j`.
    pub fn pairs_kinds(&self, kind_i: ParticleKind, kind_j: ParticleKind) -> f64 {
        if kind_i == kind_j {
            let energy = self.pairs_same_kinds(kind_i);
            return energy;
        }

        let kind_indices = self.system.particles().kind.iter().enumerate();
        let ids_i: Vec<usize> =
            kind_indices.clone().filter(|&(_, &kind)| kind == kind_i).map(|(id, _)| id).collect();
        let ids_j: Vec<usize> =
            kind_indices.filter(|&(_, &kind)| kind == kind_j).map(|(id, _)| id).collect();

        if ids_i.len() == 0 || ids_j.len() == 0 {
            return 0.0;
        };

        ids_i.iter()
            .map(|&i| {
                let mut local_energy = 0.0;
                for &j in ids_j.iter() {
                    let r = self.system.nearest_image(i, j).norm();
                    local_energy += self.pair(r, i, j);
                    println!("i: {}, j: {}, ener: {}", i, j, local_energy)
                }
                local_energy
            })
            .sum()
    }

    /// Compute the energy due to long range corrections for the pairs
    #[inline]
    pub fn pairs_tail(&self) -> f64 {
        if self.system.cell.is_infinite() {
            return 0.0;
        }
        let mut energy = 0.0;
        let volume = self.system.volume();
        let composition = self.system.composition();
        for i in self.system.particle_kinds() {
            let ni = composition[i] as f64;
            for j in self.system.particle_kinds() {
                let nj = composition[j] as f64;
                for potential in self.system.interactions().pairs(i, j) {
                    energy += 2.0 * PI * ni * nj * potential.tail_energy() / volume;
                }
            }
        }
        return energy;
    }

    /// Compute the energy due to long range corrections
    #[inline]
    pub fn pairs_kinds_tail(&self, i: ParticleKind, j: ParticleKind) -> f64 {
        if self.system.cell.is_infinite() {
            return 0.0;
        }
        let mut energy = 0.0;
        let volume = self.system.volume();
        let composition = self.system.composition();

        let ni = composition[i] as f64;
        let nj = composition[j] as f64;
        for potential in self.system.interactions().pairs(i, j) {
            energy += 2.0 * PI * ni * nj * potential.tail_energy() / volume;
        }
        return energy;
    }

    /// Compute the energy associated with the bonded particles `i, j` at
    /// distance `r`
    #[inline]
    pub fn bond(&self, r: f64, i: usize, j: usize) -> f64 {
        let mut energy = 0.0;
        for potential in self.system.bond_potentials(i, j) {
            energy += potential.energy(r);
        }
        return energy;
    }

    /// Compute the energy of all the bonds in the system
    pub fn bonds(&self) -> f64 {
        let mut energy = 0.0;
        for molecule in self.system.molecules() {
            for bond in molecule.bonds() {
                let (i, j) = (bond.i(), bond.j());
                let r = self.system.nearest_image(i, j).norm();
                energy += self.bond(r, i, j);
            }
        }
        return energy;
    }

    /// Compute the energy associated with the angle `i, j, k` at angle `theta`
    #[inline]
    pub fn angle(&self, theta: f64, i: usize, j: usize, k: usize) -> f64 {
        let mut energy = 0.0;
        for potential in self.system.angle_potentials(i, j, k) {
            energy += potential.energy(theta);
        }
        return energy;
    }

    /// Compute the energy of all the angles in the system
    pub fn angles(&self) -> f64 {
        let mut energy = 0.0;
        for molecule in self.system.molecules() {
            for angle in molecule.angles() {
                let (i, j, k) = (angle.i(), angle.j(), angle.k());
                let theta = self.system.angle(i, j, k);
                energy += self.angle(theta, i, j, k);
            }
        }
        return energy;
    }

    /// Compute the energy associated with the dihedral angle `i, j, k, m` at
    /// angle `phi`
    #[inline]
    pub fn dihedral(&self, phi: f64, i: usize, j: usize, k: usize, m: usize) -> f64 {
        let mut energy = 0.0;
        for potential in self.system.dihedral_potentials(i, j, k, m) {
            energy += potential.energy(phi);
        }
        return energy;
    }

    /// Compute the energy of all the dihedral angles in the system
    pub fn dihedrals(&self) -> f64 {
        let mut energy = 0.0;
        for molecule in self.system.molecules() {
            for dihedral in molecule.dihedrals() {
                let (i, j, k, m) = (dihedral.i(), dihedral.j(), dihedral.k(), dihedral.m());
                let phi = self.system.dihedral(i, j, k, m);
                energy += self.dihedral(phi, i, j, k, m);
            }
        }
        return energy;
    }

    /// Compute the energy of the electrostatic interactions
    #[inline]
    pub fn coulomb(&self) -> f64 {
        if let Some(coulomb) = self.system.coulomb_potential() {
            coulomb.energy(self.system)
        } else {
            0.0
        }
    }

    /// Compute the energy of the global potentials
    #[inline]
    pub fn global(&self) -> f64 {
        let mut energy = 0.0;
        for global in self.system.global_potentials() {
            energy += global.energy(self.system);
        }
        return energy;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sys::{System, UnitCell};
    use energy::{Harmonic, LennardJones, NullPotential, PairInteraction};
    use utils::{unit_from, system_from_xyz};

    fn testing_system() -> System {
        let mut system = system_from_xyz("4
        cell: 10.0
        F 0.0 0.0 0.0
        F 1.0 0.0 0.0
        F 1.0 1.0 0.0
        F 2.0 1.0 0.0
        ");
        assert!(system.add_bond(0, 1).is_empty());
        assert!(system.add_bond(1, 2).is_empty());
        assert!(system.add_bond(2, 3).is_empty());

        let mut pair = PairInteraction::new(Box::new(LennardJones {
                                                epsilon: unit_from(100.0, "kJ/mol/A^2"),
                                                sigma: unit_from(0.8, "A"),
                                            }),
                                            5.0);
        pair.enable_tail_corrections();

        system.add_pair_potential("F", "F", pair);

        system.add_bond_potential("F",
                                  "F",
                                  Box::new(Harmonic {
                                      k: unit_from(100.0, "kJ/mol/A^2"),
                                      x0: unit_from(2.0, "A"),
                                  }));

        system.add_angle_potential("F",
                                   "F",
                                   "F",
                                   Box::new(Harmonic {
                                       k: unit_from(100.0, "kJ/mol/deg^2"),
                                       x0: unit_from(88.0, "deg"),
                                   }));

        system.add_dihedral_potential("F",
                                      "F",
                                      "F",
                                      "F",
                                      Box::new(Harmonic {
                                          k: unit_from(100.0, "kJ/mol/deg^2"),
                                          x0: unit_from(185.0, "deg"),
                                      }));

        // unused interaction to check that we do handle this right
        system.add_pair_potential("H", "O", PairInteraction::new(Box::new(NullPotential), 0.0));

        return system;
    }

    #[test]
    fn pairs() {
        let system = testing_system();
        let evaluator = EnergyEvaluator::new(&system);
        assert_ulps_eq!(evaluator.pairs(), unit_from(-258.3019360389957, "kJ/mol"));
        assert_ulps_eq!(evaluator.pairs_tail(), -0.0000028110338032153973);
    }

    #[test]
    fn pairs_tail_infinite_cell() {
        let mut system = testing_system();
        system.cell = UnitCell::new();

        let evaluator = EnergyEvaluator::new(&system);
        assert_eq!(evaluator.pairs_tail(), 0.0);
    }

    #[test]
    fn bonds() {
        let system = testing_system();
        let evaluator = EnergyEvaluator::new(&system);
        assert_ulps_eq!(evaluator.bonds(), unit_from(150.0, "kJ/mol"));
    }

    #[test]
    fn angles() {
        let system = testing_system();
        let evaluator = EnergyEvaluator::new(&system);
        assert_ulps_eq!(evaluator.angles(), unit_from(400.0, "kJ/mol"));
    }

    #[test]
    fn dihedrals() {
        let system = testing_system();
        let evaluator = EnergyEvaluator::new(&system);
        assert_ulps_eq!(evaluator.dihedrals(),
                        unit_from(1250.0, "kJ/mol"),
                        max_ulps = 15);
    }

    #[test]
    fn pairs_kinds_pure() {
        let system = testing_system();
        let evaluator = EnergyEvaluator::new(&system);
        let kind = system.kind_of("F").unwrap();
        assert_eq!(evaluator.pairs(), evaluator.pairs_kinds(kind, kind));
    }

    #[test]
    fn pairs_kinds_mixture() {
        let mut system = system_from_xyz("4
        cell: 10.0
        F 0.0 0.0 0.0
        A 1.0 0.0 0.0
        F 1.0 1.0 0.0
        A 2.0 1.0 0.0
        ");

        let mut f_f = PairInteraction::new(Box::new(LennardJones {
                                               epsilon: unit_from(100.0, "kJ/mol/A^2"),
                                               sigma: unit_from(0.8, "A"),
                                           }),
                                           5.0);
        f_f.enable_tail_corrections();
        let mut a_a = PairInteraction::new(Box::new(LennardJones {
                                               epsilon: unit_from(50.0, "kJ/mol/A^2"),
                                               sigma: unit_from(0.8, "A"),
                                           }),
                                           5.0);
        a_a.enable_tail_corrections();
        let mut f_a = PairInteraction::new(Box::new(LennardJones {
                                               epsilon: unit_from(123.0, "kJ/mol/A^2"),
                                               sigma: unit_from(0.7, "A"),
                                           }),
                                           5.0);
        f_a.enable_tail_corrections();

        system.add_pair_potential("F", "A", f_a);
        system.add_pair_potential("F", "F", f_f);
        system.add_pair_potential("A", "A", a_a);
        let evaluator = EnergyEvaluator::new(&system);
        let f = system.kind_of("F").unwrap();
        let a = system.kind_of("A").unwrap();
        assert_eq!(evaluator.pairs(),
                   evaluator.pairs_kinds(a, a) + evaluator.pairs_kinds(f, f) +
                   evaluator.pairs_kinds(f, a));
        assert_eq!(evaluator.pairs_tail(),
                   evaluator.pairs_kinds_tail(a, a) + evaluator.pairs_kinds_tail(f, f) +
                   evaluator.pairs_kinds_tail(f, a) +
                   evaluator.pairs_kinds_tail(a, f));
    }
}
