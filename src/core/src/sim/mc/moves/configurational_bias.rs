// Procedures for configurational bias Monte-Carlo (CBMC)
//
// The probability of creating a trial position {b} is:
// p({b}) d{b} = p(l, theta, phi) l^2 sin(theta) dl dtheta dphi
//             = l^2 p(l) dl
//               sin(theta) p(theta) dtheta
//               p(phi) dphi
//
// where
// {b}: trial position (vector)
// l: bond length
// theta: bond angle
// phi: dihedral angle
// p(.): probability density
//
// CBMC attempts to create trial positions sampling values for
// l, theta and phi with high acceptance.
//
// # Note:
//
// For the initial implementation we will assume that bond stretching and
// angle bending potentials are harmonic functions.
// The resulting Boltzmann factor is _similar_ to a Gaussian distribution.
// We can use this similarity to improve the sampling - using rejection sampling -
// of the bond length as well as the bond angle.
// Naive sampling (random sampling on a sphere) is slower by a factor ~ 2-10x.
// See the work of Vlugt et al. "Improving the efficiency of the configurational-bias
// Monte Carlo algorithm", Molecular Physics, 1998, Vol. 94, No. 4, 727-733
//
// In the future, we might implement a different sampling scheme for arbitrary potentials.

use std::f64::consts::PI;
use rand::distributions::{Sample, Normal, Range};
use rand::Rng;

use energy::{AnglePotential, DihedralPotential, Harmonic};
use types::Vector3D;

/// Creates a bond length given a harmonic bond potential.
///
/// The probability of the length is proportional to
/// the Boltzmann distribution of a harmonic bond potential:
///
/// p(l) ~ l^2 * exp[-beta * u_bond(l)] d(l)
///
// Why is dynamic dispatch not working? I.e. putting rng: &mut Box<Rng> as fn arg??
pub fn create_bond_length<T: Rng>(beta: f64, harmonic_potential: &Harmonic, rng: &mut Box<T>) -> f64 {
    // standard deviation of the gaussian distribution
    let sigma = (1f64 / (beta * harmonic_potential.k)).sqrt();
    // compute factor for rejection sampling so that
    // p(l) <= normal(l)*M
    // Taken from Frenkel, Smit
    let M = (harmonic_potential.x0 + 3.0 * sigma).powi(2);
    // create bond length according to normal distribution
    let mut normal = Normal::new(harmonic_potential.x0, sigma);

    // use rejection sampling to sample from non-gaussian distribution density
    let mut l = 0f64;
    loop {
        l = normal.sample(rng);
        // accept/reject
        if rng.next_f64() <= l * l / M { break }
    }
    l // return length
}

/// Creates a angle given a harmonic bond angle potential.
///
/// p(theta) ~ sin(theta) * exp[-beta * u_angle(theta)] d(theta)
///
/// The position is created with respect to two neighboring positions.
/// The probability for the angle is proportional to
/// the Boltzmann distribution of the angle potential.
pub fn create_bond_angle<T: Rng>(
    beta: f64,
    harmonic_potential: &Harmonic,
    rng: &mut Box<T>
) -> f64 {
    // standard deviation of the gaussian distribution
    let sigma = (1f64 / (beta * harmonic_potential.k)).sqrt();
    // create bond length according to normal distribution
    let mut normal = Normal::new(harmonic_potential.x0, sigma);

    // use rejection sampling to sample from non-gaussian distribution density
    let mut theta = 0f64;
    loop {
        theta = normal.sample(rng);
        // accept/reject
        if rng.next_f64() <= theta.sin() { break }
    }
    theta // return angle
}

/// Creates a bond angle `theta` and a dihedral angle `phi`.
pub fn create_bond_and_dihedral_angle<T: Rng>(
    beta: f64,
    harmonic_potential: &Harmonic,
    dihedral_potential: &Box<DihedralPotential>,
    rng: &mut Box<T>
) -> (f64, f64) {

    // Sampling range for angle
    let mut phi_range = Range::new(0.0, 2.0 * PI);
    let mut phi = 0f64;
    let mut theta = 0f64;
    loop {
        // Create angle according to harmonic potential
        theta = create_bond_angle(beta, harmonic_potential, rng);
        let bending_energy = harmonic_potential.k * 0.5 * (theta - harmonic_potential.x0).powi(2);
        // Create point on a cone with angle theta
        phi = phi_range.sample(rng);
        let dihedral_energy = dihedral_potential.energy(phi);
        if rng.next_f64() <= f64::exp(-beta * (bending_energy + dihedral_energy)) { break }
    }
    (theta, phi) // return angles
}

/// Creates a new position given an arbitrary bond angle potential.
///
/// The position is created with respect to two neighboring positions.
/// The probability for the angle is proportional to
/// the Boltzmann distribution of the angle potential.
// pub fn position_from_angle_arbitrary_potential<T: Rng>(
//     positions: &[Vector3D; 2],
//     beta: f64,
//     angle_potential: &Box<AnglePotential>,
//     rng: &mut Box<T>
// ) -> Vector3D {
//     let mut normal = Normal::new(0f64, 1f64);

//     let r10 = (positions[1] - positions[0]).normalized();
//     let mut theta = 0f64;

//     loop {
//         // Create new vector uniformly distributed on a sphere
//         let r12 = Vector3D::new(
//             normal.sample(rng),
//             normal.sample(rng),
//             normal.sample(rng)
//         ).normalized();
//         // Compute angle between bond1 and bond2
//         theta = f64::acos(r10 * r12);
//         // naive implementation, optimize?
//         if rng.next_f64() <= f64::exp(-beta * angle_potential.energy(theta)) { break }
//     }
//     theta
// }

/// Create a new position given a dihedral and angle bending potential.
///
/// The new position is created with respect to three neighboring positions.
/// The distributions of the dihedral and bonding angle are proportional
/// to the Boltzmann factor exp(-beta * [u_angle + u_dihedral])
// pub fn position_from_angle_and_dihedral<T: Rng>(
//     positions: &[Vector3D; 3],
//     beta: f64,
//     angle_potential: &Box<AnglePotential>,
//     dihedral_potential: &Box<DihedralPotential>,
//     rng: &mut Box<T>
// ) -> Vector3D {

//     let mut normal = Normal::new(0f64, 1f64);

//     // compute dihedral angle
//     let r01 = (positions[1] - positions[0]).normalized();
//     let r12 = (positions[2] - positions[1]).normalized();

//     // dihedral angle
//     let mut phi = 0f64;
//     loop {
//         // Create new vector uniformly distributed on a sphere
//         let r32 = Vector3D::new(
//             normal.sample(rng),
//             normal.sample(rng),
//             normal.sample(rng)
//         ).normalized();
//         let theta = f64::acos(r12 * r32);
//         let bending_energy = angle_potential.energy(theta);

//         let u = r01 ^ r12;
//         let v = r12 ^ r32;
//         phi = f64::acos(u * v);
//         let dihedral_energy = dihedral_potential.energy(phi);
//         if rng.next_f64() <= f64::exp(-beta * (bending_energy + dihedral_energy)) { break }
//     }
//     phi // return angle
// }

#[cfg(test)]
mod tests {
    use super::*;
    use energy::Harmonic;
    use std::f64::consts::PI;
    use rand::{XorShiftRng, SeedableRng, Rng};
    use consts::K_BOLTZMANN;

    #[test]
    fn test_create_bond_length() {
        let mut rng = Box::new(XorShiftRng::new_unseeded());
        rng.reseed([2015u32, 42u32, 3u32, 12u32]);
        let beta = 1.0 / (K_BOLTZMANN * 50.0);
        let harmonic_potential = Harmonic{k: 62500. * K_BOLTZMANN, x0: 1.540};
        let iterations = 5000;
        let l: f64 = (0..iterations).map(|_| create_bond_length(beta, &harmonic_potential, &mut rng)).sum();
        assert!(f64::abs((l/iterations as f64 - harmonic_potential.x0)/harmonic_potential.x0) < 1e-3)
    }

    #[test]
    fn test_create_bond_angle() {
        let mut rng = Box::new(XorShiftRng::new_unseeded());
        rng.reseed([2015u32, 42u32, 3u32, 12u32]);
        let beta = 1.0 / (K_BOLTZMANN * 50.0);
        let harmonic_potential = Harmonic{k: 62500. * K_BOLTZMANN, x0: f64::to_radians(114.0)};
        let iterations = 1000;
        let theta: f64 = (0..iterations).map(|_| create_bond_angle(beta, &harmonic_potential, &mut rng)).sum();
        assert!(f64::abs((theta/iterations as f64 - harmonic_potential.x0)/harmonic_potential.x0) < 1e-3)
    }

    // #[test]
    // fn test_create_bond_and_dihedral_angle() {
    //     unimplemented!();
    // }
}
