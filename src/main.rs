use machine_learning::*;
use std::io::{self};

fn main() -> io::Result<()> {
    println!("Machine learning.. Auto clasification");

    let data = vec![
        (Auto::new(100, 2), Coefficient(0, 0)),
        (Auto::new(500, 4), Coefficient(1, 1)),
        (Auto::new(200, 2), Coefficient(0, 0)),
        (Auto::new(500, 2), Coefficient(1, 0)),
        (Auto::new(500, 6), Coefficient(1, 1)),
        (Auto::new(200, 5), Coefficient(0, 1)),
        (Auto::new(350, 4), Coefficient(0, 1)),
        (Auto::new(250, 3), Coefficient(0, 0)),
        (Auto::new(400, 3), Coefficient(1, 0)),
        (Auto::new(450, 4), Coefficient(1, 1)),
    ];

    let robot = Robot::new(&data).train();
    let mut buffer = String::new();

    print_expl();

    while !buffer.contains(&String::from("exit!"))  {
        buffer = String::new();
        io::stdin().read_line(&mut buffer)?;
        let parts: Vec<&str> = buffer.trim().split(" ").collect::<Vec<&str>>();

        print_expl();

        let cargo = match parts.get(0).unwrap().parse::<u32>() {
            Ok(val) => val,
            _ => {
                println!("Invalid value for cargo");
                continue;
            }
        };

        let seats = match parts.get(1).unwrap_or(&"").parse::<u8>() {
            Ok(val) => val,
            _ => {
                println!("Invalid value for seats");
                continue;
            }
        };

        let class = robot.guess(&Auto::new(cargo, seats));

        if let None = class {
            println!("Auto doesn't match a patterns");
        } else {
            println!("{}", class.unwrap());
        }
    }

    Ok(())
}

fn print_expl() {
    println!("Enter new auto: {{cargo:u32}} {{seats:u8}}");
}
