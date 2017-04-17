// Lumol, an extensible molecular simulation engine
// Copyright (C) 2015-2016 G. Fraux — BSD license

use rand::distributions::{Sample, Range};
use rand::Rng;

use std::usize;

use super::MCMove;
use super::select_molecule;

use types::Vector3D;
use sys::{System, Molecule, EnergyCache};

/// Monte-Carlo move for rebuilding a molecule
pub struct Rebuild {
    /// Type of molecule to Rebuild. `None` means all molecules.
    moltype: Option<u64>,
    /// Index of the molecule to rebuild
    molid: usize,
    /// Vector that keeps track of which segments are already present in the new molecule
    is_segment_present: Vec<bool>,
    /// New positions of the rebuilt molecule
    new_pos: Vec<Vector3D>,
    /// Rosenbluth weight of the new configuration
    new_weight: Vec<f64>,
    /// Rosenbluth weight of the old configuration
    old_weight: Vec<f64>,
    /// number of configurational bias steps
    k: u64,
}

impl Rebuild {
    /// Create a new `Rebuild` move, using `k`configurational bias steps.
    pub fn new(k: u64) -> Rebuild {
        Rebuild::create(k, None)
    }

    /// Create a new `Rebuild` move, using `k`configurational bias steps.
    /// Rebuilding only molecules of type `moltype`.
    pub fn with_moltype(k: u64, moltype: u64) -> Rebuild {
        Rebuild::create(k, Some(moltype))
    }

    /// Factorizing the constructors
    fn create(k: u64, moltype: Option<u64>) -> Rebuild {
        Rebuild {
            moltype: moltype,
            molid: usize::MAX,
            molecule: Vec::new(),
            newpos: Vec::new(),
            k: k,
        }
    }
}

impl Default for Rebuild {
    fn default() -> Rebuild {
        Rebuild::new(1)
    }
}

impl MCMove for Rebuild {
    fn describe(&self) -> &str {
        "rebuilding a molecule"
    }

    fn setup(&mut self, _: &System) {
    }

    fn prepare(&mut self, system: &mut System, rng: &mut Box<Rng>) -> bool {
        if let Some(id) = select_molecule(system, self.moltype, rng) {
            self.molid = id;
        } else {
            warn!("Cannot rebuild molecule: no molecule of this type in the system.");
            return false;
        }

        let ids: Vec<usize> = system.molecule(self.molid).iter().collect();

        // rebuild the molecule starting from the 2nd (index 1) segment
        for segment_id, id in ids.enumerate() {
            // generate trial segments for the new configuration
            let mut boltzmann_weight: Vec<f64> = Vec::new();
            for i in 0..self.k {
                // if only bond -> create from bond
                // if bond + angle -> create from angle with bond length from bond
                // if dihedreal -> whole procedure
                let external_energy = system.particle_energy(selected_segment);
                boltmann_weight.push()
            }
        }
        // generate all trial positions:

        // pick first bead and select a position
        // grow next bead
        // 1. pick position on sphere
        // 2. pick length
        // 3. compute contributions due to bond length, angle and torsion

        return true;
    }

    fn cost(&self, system: &System, beta: f64, cache: &mut EnergyCache) -> f64 {
        let idxes = system.molecule(self.molid).iter().collect::<Vec<_>>();
        let cost = cache.move_particles_cost(system, idxes, &self.newpos);
        return cost*beta;
    }

    fn apply(&mut self, system: &mut System) {
        for (i, pi) in system.molecule(self.molid).iter().enumerate() {
            system[pi].position = self.newpos[i];
        }
    }

    fn restore(&mut self, _: &mut System) {
        // Nothing to do.
    }

    fn update_amplitude(&mut self, scaling_factor: Option<f64>) {
        // Nothing to do.
    }
}

/// Computes the intramolecular energy between a segment and all other segments
/// in the (fractional) molecule.
fn segment_noncovalent_intramolecular_energy(
    segment_id: usize,
    molecule_positions: &Vec<Vector3D>,
    molecule: &Molecule,
) -> f64 {
    let mut energy = 0.0;
    // loop over all
    let segment_position = molecule_positions[segment_id];
    for i, position in molecule_positions.enumerate() {
        let r = (position - segment_position).norm();
        for potential in system.pair_potentials(segment_id, i) {
            let info = potential.restriction().information(-1);
            if !info.excluded {
                energy += info.scaling * potential.energy(r);
            }
        }
    }
}

/// Computes the intermolecular energy between a segment and all other atoms in the system
/// that are not part of the same molecule.
fn segment_intermolecular_energy(
    system: &System,
    segment_position: &Vector3D,
    segment_id: usize
) -> f64 {
    let mut energy = 0.0;
    let cell = system.cell();
    // maybe rewrite this loop to go over molecules
    for i in 0..self.system.size() {
        if system.are_in_same_molecule(i, segment_id) { break };
        let r = cell.distance(segment_position, system[i].position);
        for potential in system.pair_potentials(segment_id, i) {
            let info = potential.restriction().information(-1);
            if !info.excluded {
                energy += info.scaling * potential.energy(r);
            }
        }
    }
    // TODO: add global interactions
    energy
}
