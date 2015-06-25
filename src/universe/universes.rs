/*
 * Cymbalum, Molecular Simulation in Rust
 * Copyright (C) 2015 Guillaume Fraux
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/
*/

use std::collections::HashMap;
use std::ops::{Index, IndexMut};

use ::potentials::PairPotential;
use ::types::Vector3D;

use super::Particle;
use super::UnitCell;
use super::interactions::Interactions;

/// The Universe type hold all the data about a system.
pub struct Universe {
    /// List of particles in the system
    particles: Vec<Particle>,
    /// Particles types, associating particles names and indexes
    types: HashMap<String, usize>,
    /// Interactions is a hash map associating particles types and potentials
    interactions: Interactions,
    /// Unit cell of the universe
    cell: UnitCell,
}

impl Universe {
    /// Create a new empty Universe
    pub fn new() -> Universe {
        Universe{
            particles: Vec::new(),
            types: HashMap::new(),
            interactions: Interactions::new(),
            cell: UnitCell::new(),
        }
    }

    /// Create an empty universe with a specific UnitCell
    pub fn from_cell(cell: UnitCell) -> Universe {
        let mut universe = Universe::new();
        universe.set_cell(cell);
        return universe;
    }

    /// Get the universe unit cell
    pub fn cell<'a>(&'a self) -> &'a UnitCell {&self.cell}
    /// Set the universe unit cell
    pub fn set_cell(&mut self, cell: UnitCell) {self.cell = cell;}

    /// Insert a particle at the end of the internal list
    pub fn add_particle(&mut self, p: Particle) {self.particles.push(p);}
    /// Get the number of particles in this universe
    pub fn size(&self) -> usize {self.particles.len()}

    /// Get the list of pair interaction betweent the atom i and the atom j
    pub fn pairs<'a>(&'a self, i: usize, j: usize) -> &'a Vec<Box<PairPotential>> {
        let itype = self.types[self.particles[i].name()];
        let jtype = self.types[self.particles[j].name()];
        &self.interactions.pairs[&(itype, jtype)]
    }

    /// Add an interaction between the particles with names `names`
    pub fn add_pair_interaction<T>(&mut self, i: &str, j: &str, pot: T) where T: PairPotential + 'static {
        let itype = self.get_type(i);
        let jtype = self.get_type(j);

        if !self.interactions.pairs.contains_key(&(itype, jtype)) {
            self.interactions.pairs.insert((itype, jtype), Vec::new());
        }
        let pairs = self.interactions.pairs.get_mut(&(itype, jtype)).unwrap();
        pairs.push(Box::new(pot));
    }

    /// Get the distance between the particles at indexes `i` and `j`
    pub fn distance(&self, i: usize, j:usize) -> f64 {
        self.cell.distance(self.particles[i].position(), self.particles[j].position())
    }

    /// Get or create the usize type for the particle `name`
    fn get_type(&mut self, name: &str) -> usize {
        if self.types.contains_key(name) {
            self.types[name]
        } else {
            let index = self.types.len();
            self.types.insert(name.to_string(), index);
            return index;
        }
    }
}

impl Index<usize> for Universe {
    type Output = Particle;
    fn index<'a>(&'a self, index: usize) -> &'a Particle {
        &self.particles[index]
    }
}

impl IndexMut<usize> for Universe {
    #[inline]
    fn index_mut<'a>(&'a mut self, index: usize) -> &'a mut Particle {
        &mut self.particles[index]
    }
}

#[cfg(test)]
mod tests {
    use ::universe::*;
    use ::types::*;
    use ::potentials::*;

    #[test]
    fn particles() {
        let mut universe = Universe::new();
        universe.add_particle(Particle::new("O"));
        universe.add_particle(Particle::new("H"));
        universe.add_particle(Particle::new("H"));

        assert_eq!(universe.size(), 3);
        assert_eq!(universe[0].name(), "O");
        assert_eq!(universe[1].name(), "H");
        assert_eq!(universe[2].name(), "H");
    }

    #[test]
    fn distances() {
        let mut universe = Universe::from_cell(UnitCell::cubic(5.0));
        universe.add_particle(Particle::new("O"));
        universe.add_particle(Particle::new("H"));

        universe[0].set_position(Vector3D::new(9.0, 0.0, 0.0));
        universe[1].set_position(Vector3D::new(0.0, 0.0, 0.0));
        assert_eq!(universe.distance(0, 1), 1.0);

        universe.set_cell(UnitCell::new());
        assert_eq!(universe.distance(0, 1), 9.0);
    }

    #[test]
    fn pairs() {
        let mut universe = Universe::new();
        universe.add_particle(Particle::new("He"));

        universe.add_pair_interaction("He", "He", LennardJones{sigma: 0.3, epsilon: 2.0});
        universe.add_pair_interaction("He", "He", LennardJones{sigma: 0.6, epsilon: 2.0});

        assert_eq!(universe.pairs(0, 0).len(), 2);
    }
}
