pub mod command_line {
    use std::collections::HashMap;

    const PREFIX: &str = "--";
    const PROGRAM: &str = "--program";

    pub fn parse_args(args: std::env::Args, implicit_first_key: Option<String>) -> Result<HashMap<String, Option<String>>, String> {
        let mut map = HashMap::new();

        let mut enumerator = args.enumerate().peekable();

        while let Some((ndx, key)) = enumerator.next() {
            //0th arg is the program path
            if ndx == 0 {
                map.insert(String::from(PROGRAM), Some(key));
                continue;
            }

            //If first argument has implicit key, use this as value
            if ndx == 1 && !key.starts_with(PREFIX) {
                if let Some(ref first_key) = implicit_first_key {
                    map.insert(first_key.clone(), Some(key));
                    continue;
                }
            }

            //Check if valid arg
            if !key.starts_with(PREFIX) {
                return Err(format!("Invalid argument {key}"));
            }

            //Try get value
            if let Some((_, value)) = enumerator.peek() {
                //Don't use next key as value
                if value.starts_with(PREFIX) {
                    map.insert(key, None);
                }
                else {
                    map.insert(key, Some(value.clone()));
                    _ = enumerator.next();
                }
            }
            else {
                map.insert(key, None);
            }
        }

        Ok(map)
    }
}