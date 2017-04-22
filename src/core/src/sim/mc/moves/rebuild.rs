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
    /// New positions of the rebuilt molecule
    newpos: Vec<Vector3D>,
    /// Rosenbluth weight of the new configuration
    new_weights: Vec<f64>,
    /// Rosenbluth weight of the old configuration
    old_weight: f64,
    /// number of configurational bias steps
    cb_steps: u64,
    /// early rejection tolerance, hardcode?
    early_rejection_tolerance: f64,
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
            early_rejection_tolerance: 1.0e-200,
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

        // clear
        self.newpos.clear();
        self.new_weights.clear();
        self.old_weight = 0.0;

        let mut trial_weights: Vec<f64> = Vec::with_capacity(self.cb_steps);
        let mut trialpos: Vec<Vector3D> = Vec::with_capacity(self.cb_steps);

        let beta = 1.0 / (K_BOLTZMANN * system.temperature());

        // loop over each atom (segment) of the molecule
        // .iter() returns the index of the atom to use to index into system
        // pid == particle index
        for pid in system.molecule(self.molid).iter() {
            trial_weights.clear();
            trialpos.clear();

            // loop over trials
            // this is how you'd find it in literature
            // maybe it is better to lift loop into `trial_position` and `compute_particle_weight`?
            // that would save us all (but one) lookup in trial_position
            // i.e write:
            // fn trial_positions(system, pid, self.newpos, beta, rng) -> &Vec<Vector3D>
            // where the resulting array would contain cb_steps elements
            for step in 0..self.cb_steps {
                // new weight
                trialpos.push(trial_position(system, pid, &self.newpos, &beta, rng));
                // energy between the trial particle and ALL other partiles (excluding the growing molecule)
                // the function name is missleading, it only computes the BMF, i.e. exp(-beta U)
                trial_weights.push(compute_particle_weight(system, &trialpos, pid));
            };
            let sum_of_weights = trial_weights.iter().sum();
            trial_weights /= sum_of_weights;  // will not work, do this element wise (map)
            self.new_weights.push(sum_of_weights);
            self.newpos.push(select_position(&trialpos, &trial_weights, rng));

            // compute weight for OLD configuration
            // the zero'th weight is the old position
            // Q: use one loop.
            // Q: use existing configs instead of random sampling?
            trial_weights.push(compute_particle_weight(system, system[pid].position, pid));
            for step in 1..self.cb_steps {
                // new weight
                // do the same as above m.b. refactor into fn
            };
            self.old_weights.push(trial_weights.iter().sum())
        };
        true
    }

    fn cost(&self, system: &System, beta: f64, cache: &mut EnergyCache) -> f64 {
        let idxes = system.molecule(self.molid).iter().collect::<Vec<_>>();
        // update cache
        let _ = cache.move_particles_cost(system, idxes, &self.newpos);
        let new_weight = self.new_weights.product();
        let old_weight = self.old_weights.product();
        return beta * (new_weight / old_weight);
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
        // We could check the scaling and increase the number of steps
        // if the scaling is larger than one or decrease if its lower.
        if let Some(s) = scaling_factor {
            match s {
              s < 1.0  =>,
              _ =>,
            }
        }
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
