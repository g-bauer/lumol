// Lumol, an extensible molecular simulation engine
// Copyright (C) 2015-2016 G. Fraux — BSD license

//! Various internal utilities, which do not have there own module
#[macro_use]
mod macros;

mod alternator;
#[cfg(test)]
mod xyz;
#[cfg(test)]
pub use self::xyz::system_from_xyz;

pub use self::alternator::Alternator;

/// Internal version of `units::from`, where the unit is assumed to be correct
pub fn unit_from(value: f64, unit: &str) -> f64 {
    ::units::from(value, unit).expect("Internal unit error. This is a bug.")
}

/// Internal version of `units::to`, where the unit is assumed to be correct
pub fn unit_to(value: f64, unit: &str) -> f64 {
    ::units::to(value, unit).expect("Internal unit error. This is a bug.")
}
