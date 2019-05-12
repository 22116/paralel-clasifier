extern crate rand;
extern crate ndarray;


mod math;

use std::fmt;
use crate::math::weights::Weight;

pub type Data = [(Auto, Coefficient)];

#[derive(Clone)]
pub struct Robot<'a> {
    data: &'a Data,
    w0: Weight,
    w: [Weight; 2],
}

#[derive(Debug)]
pub enum Class {
    Light,
    Heavy,
}

#[derive(Debug, Clone)]
pub struct Coefficient(pub u8, pub u8);

#[derive(Debug, Default, Clone)]
pub struct Auto {
    pub cargo: f32,
    pub seats: f32,
}

impl Auto {
    pub fn new(cargo: u32, seats: u8) -> Self {
        Self { cargo: cargo as f32, seats: seats as f32 }
    }
}

impl fmt::Display for Class {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "{}", match self {
            Class::Heavy => "Heavy",
            Class::Light => "Lightwieght",
        })
    }
}

impl<'a> Robot<'a> {
    pub fn new(data: &'a Data) -> Self {
        Self { data, w0: Default::default(), w: Default::default() }
    }

    pub fn train(mut self) -> Self {
        let (w0, w) = math::weights::calculate(&math::norm(&self.data));

        self.w0 = w0;
        self.w = w;

        self.clone()
    }

    pub fn guess(&self, auto: &Auto) -> Option<Class> {
        let mut auto = auto.to_owned();
        let (max_cargo, max_seats) = math::max(&self.data);

        auto.cargo /= max_cargo as f32;
        auto.seats /= max_seats as f32;

        let result = math::weights::clasification(&self.w0, &self.w, &auto);

        println!("{:?}, {:?}", self.w0, self.w);
        println!("{:?}", result);

        match result {
            [Some(a), Some(b)] if a == 0.0 && b == 0.0 => Some(Class::Light),
            [Some(a), Some(b)] if a == 1.0 && b == 1.0 => Some(Class::Heavy),
            _ => None
        }
    }
}
