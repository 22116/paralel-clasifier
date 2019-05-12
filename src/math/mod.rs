use crate::{Data, Auto, Coefficient};
use std::cmp;

pub mod weights;

/// Normalize Auto dataset
pub fn norm(data: &Data) -> Vec<(Auto, Coefficient)> {
    let mut data = data.to_vec();
    let max = max(&data);

    for (a, _c) in data.iter_mut() {
        (*a).cargo /= max.0 as f32;
        (*a).seats /= max.1 as f32;
    }

    data
}

/// Maximal values for Auto dataset
pub fn max(data: &Data) -> (u32, u32) {
    let mut max1: u32 = 0;
    let mut max2: u32 = 0;

    for (a, _c) in data.iter() {
        max1 = cmp::max(max1, a.cargo as u32);
        max2 = cmp::max(max2, a.seats as u32);
    }

    (max1, max2)
}

trait Max {
    type M;

    fn max(&self) -> Self::M;
}

impl Max for Auto {
    type M = u32;

    fn max(&self) -> Self::M {
        cmp::max(self.seats as u32, self.cargo as u32)
    }
}
