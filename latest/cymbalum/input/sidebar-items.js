initSidebarItems({"enum":[["Error","Possible causes of error when reading input files"]],"fn":[["guess_bonds","Guess the bonds in a system"],["read_config","Read a whole simulation input file."],["read_interactions","Read interactions from the TOML file at `path`, and add them to the `system`. For a full documentation of the input files syntax, see the user manual."],["read_molecule","Read a the first molecule from the file at `path`. If no bond information exists in the file, bonds are guessed."]],"struct":[["Trajectory","A Trajectory is a file containing one or more successives simulation steps"],["TrajectoryError","Possible error when reading and wrinting to trajectories"]],"trait":[["FromToml","Convert a TOML table to a Rust type."],["FromTomlWithData","Convert a TOML table and some additional data to a Rust type."],["FromTomlWithPairs","Convert a TOML table and a `PairPotential` to a Rust type. This is intended to be used by potential computation mainly."]],"type":[["Result","Custom `Result` type for input files"]]});