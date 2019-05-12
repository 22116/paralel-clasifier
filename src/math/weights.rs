use rand::prelude::*;
use super::{Data, Auto};
use crate::Coefficient;
use arrayvec::ArrayVec;
use ndarray::{arr2, arr1};

pub type Weight = [Option<f32>; 2];

/// Calculate weights from training data
pub fn calculate(data: &Data) -> (Weight, [Weight; 2]) {
    let (mut w0, mut w) = get_weights();
    let mut sum = 0;

    while sum < data.len() {
        sum = 0;
        for (a, c) in data.iter() {
            let y = clasification(&w0, &w, &a);

            let err = c.as_weight().div(&y);

            match err {
                [Some(a), Some(b)] if a == 0.0 && b == 0.0 => sum += 1,
                _ => sum = 0,
            }

            w = w.add(&err.delta(&a.as_weight()));
            w0 = w0.add(&err);
        }
    }

    (w0, w)
}

/// Get Y value: `Y = f(W * X + W0)`
pub fn clasification(w0: &Weight, w: &[Weight; 2], a: &Auto) -> Weight {
    normalization(&w.mul(&a.as_weight()).add(&w0))
}

fn normalization(w: &Weight) -> Weight {
    let mut w = w.to_owned();

    for val in w.iter_mut() {
        *val = Some(match val.unwrap() as f32 {
            v if v >= 0.0 => 1.0,
            _ => 0.0
        });
    }

    w
}

fn get_weights() -> (Weight, [Weight; 2]) {
    (get_w0(), get_w())
}

fn get_w() -> [Weight; 2] {
    let mut w: [Weight; 2] = Default::default();

    for row in w.iter_mut() {
        for cell in row.iter_mut() {
            *cell = Some(rand::thread_rng().gen_range(-0.3, 0.3));
        }
    }

    w
}

fn get_w0() -> Weight {
    let mut w0: Weight = Default::default();

    for val in w0.iter_mut() {
        *val = Some(rand::thread_rng().gen_range(-0.3, 0.3));
    }

    w0
}

trait AsArray {
    type A;

    fn as_array(&self) -> Self::A;
}

trait WeightAdd {
    type R;

    fn add(&self, new: &Self::R) -> Self;
}

trait Mul {
    fn mul(&self, a: &Weight) -> Weight;
}

impl Mul for [Weight; 2] {
    fn mul(&self, a: &Weight) -> Weight {
        let data = arr2(&self.to_owned().as_array());
        let a = arr1(&a.to_owned().as_array());

        let res = data.dot(&a);

        [Some(*res.get(0).unwrap()), Some(*res.get(1).unwrap())]
    }
}

impl WeightAdd for [Weight; 2] {
    type R = [Weight; 2];

    fn add(&self, new: &Self::R) -> Self {
        let mut data = self.to_owned();

        for (i, row) in data.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = Some(new[i][j].unwrap() + cell.unwrap());
            }
        }

        data
    }
}

impl AsArray for [Weight; 2] {
    type A = [[f32; 2]; 2];

    fn as_array(&self) -> Self::A {
        let a: ArrayVec<[_; 2]> = self.to_owned().iter_mut().map(|el| {
            let a: ArrayVec<[_; 2]> = el.iter_mut().map(|el| {
                el.unwrap()
            }).collect();

            a.into_inner().unwrap()
        }).collect();

        a.into_inner().unwrap()
    }
}

impl WeightAdd for Weight {
    type R = Self;

    fn add(&self, new: &Self::R) -> Self {
        [
            Some(self.get(0).unwrap().unwrap() + new.get(0).unwrap().unwrap()),
            Some(self.get(1).unwrap().unwrap() + new.get(1).unwrap().unwrap()),
        ]
    }
}

trait WeightBasic {
    type R;

    fn delta(&self, x: &Self) -> Self::R;
    fn div(&self, x: &Self) -> Self;
    fn is_null(&self) -> bool;
}

impl WeightBasic for Weight {
    type R = [Weight; 2];

    fn delta(&self, x: &Self) -> Self::R {
        let mut data: [Weight; 2] = Default::default();

        for (i, row) in data.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = Some(self[i].unwrap() * x[j].unwrap());
            }
        }

        data
    }

    fn div(&self, x: &Self) -> Self {
        [
            Some(self.get(0).unwrap().unwrap() - x.get(0).unwrap().unwrap()),
            Some(self.get(1).unwrap().unwrap() - x.get(1).unwrap().unwrap()),
        ]
    }

    fn is_null(&self) -> bool {
        if let [Some(a), Some(b)] = self {
            if *a == 0.0 as f32 && *b == 0.0 as f32 {
                return true;
            }
        }

        return false;
    }
}

trait AsWeight {
    fn as_weight(&self) -> Weight;
}

impl AsWeight for Auto {
    fn as_weight(&self) -> Weight {
        [Some(self.cargo), Some(self.seats)]
    }
}

impl AsArray for Auto {
    type A = [f32; 2];

    fn as_array(&self) -> Self::A {
        [self.cargo, self.seats]
    }
}

impl AsWeight for Coefficient {
    fn as_weight(&self) -> Weight {
        [Some(self.0 as f32), Some(self.1 as f32)]
    }
}

impl AsArray for Weight {
    type A = [f32; 2];

    fn as_array(&self) -> Self::A {
        [self[0].unwrap(), self[1].unwrap()]
    }
}

impl AsArray for Coefficient {
    type A = [f32; 2];

    fn as_array(&self) -> Self::A {
        [self.0 as f32, self.1 as f32]
    }
}
