use crate::obsidian::{GetLinkedNotesRequest, Obsidian, SearchWithContextRequest};

#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_example_vault() -> TempDir {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
        let vault_path = temp_dir.path();

        // Create campaign overview file
        let campaign_file = vault_path.join("campaign_overview.md");
        let mut file = fs::File::create(&campaign_file).unwrap();
        writeln!(file, "# The Lost Crown Campaign").unwrap();
        writeln!(file, "").unwrap();
        writeln!(
            file,
            "This campaign follows adventurers searching for the Crown of Aldric."
        )
        .unwrap();
        writeln!(file, "").unwrap();
        writeln!(file, "## Key Locations").unwrap();
        writeln!(file, "- [[Vaelthorne City]] - The capital").unwrap();
        writeln!(file, "- [[The Whispering Woods]] - Mysterious forest").unwrap();
        writeln!(file, "").unwrap();
        writeln!(file, "## Major NPCs").unwrap();
        writeln!(file, "- [[Lord Commander Marcus]] - City guard leader").unwrap();

        // Create NPC file that links back
        let npc_file = vault_path.join("Lord Commander Marcus.md");
        let mut file = fs::File::create(&npc_file).unwrap();
        writeln!(file, "# Lord Commander Marcus").unwrap();
        writeln!(file, "").unwrap();
        writeln!(
            file,
            "A trusted ally in [[campaign_overview|The Lost Crown Campaign]]."
        )
        .unwrap();
        writeln!(
            file,
            "He serves as commander of the guard in [[Vaelthorne City]]."
        )
        .unwrap();
        writeln!(file, "").unwrap();
        writeln!(
            file,
            "Marcus has been investigating the Crown of Aldric mystery."
        )
        .unwrap();

        // Create a file with no links
        let rules_file = vault_path.join("Combat Rules.md");
        let mut file = fs::File::create(&rules_file).unwrap();
        writeln!(file, "# Combat Rules Reference").unwrap();
        writeln!(file, "").unwrap();
        writeln!(file, "Initiative: Roll 1d20 + Dex modifier").unwrap();
        writeln!(file, "Attack: Roll 1d20 + BAB + ability modifier").unwrap();
        writeln!(file, "Damage: Roll weapon damage + ability modifier").unwrap();

        temp_dir
    }

    #[test]
    fn test_search_with_context_integration() {
        let temp_dir = create_example_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Test basic search
        let request = SearchWithContextRequest {
            query: "Crown of Aldric".to_string(),
            context_lines: Some(1),
            regex: Some(false),
            case_sensitive: Some(false),
        };

        let result = obsidian.search_with_context(request).unwrap();

        // Should find matches in both files
        let content_str = format!("{:?}", result.content[0]);
        assert!(content_str.contains("Crown of Aldric"));
        assert!(
            content_str.contains("campaign_overview.md")
                || content_str.contains("Lord Commander Marcus.md")
        );

        println!("Search with context test passed!");
    }

    #[test]
    fn test_search_with_context_regex() {
        let temp_dir = create_example_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Test regex search for dice notation
        let request = SearchWithContextRequest {
            query: r"1d20".to_string(),
            context_lines: Some(1),
            regex: Some(true),
            case_sensitive: Some(false),
        };

        let result = obsidian.search_with_context(request).unwrap();

        let content_str = format!("{:?}", result.content[0]);
        assert!(content_str.contains("1d20"));
        assert!(content_str.contains("Combat Rules.md"));

        println!("Regex search test passed!");
    }

    #[test]
    fn test_get_linked_notes_integration() {
        let temp_dir = create_example_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Test getting links for campaign overview
        let request = GetLinkedNotesRequest {
            filename: "campaign_overview.md".to_string(),
        };

        let result = obsidian.get_linked_notes(request).unwrap();

        let content_str = format!("{:?}", result.content[0]);

        assert!(content_str.contains("campaign_overview.md"));
        assert!(content_str.contains("outgoing_links"));
        assert!(content_str.contains("incoming_links"));

        // Should have outgoing links to locations and NPCs
        assert!(
            content_str.contains("Vaelthorne City")
                || content_str.contains("Lord Commander Marcus")
        );

        // Should have incoming link from the NPC file
        assert!(content_str.contains("Lord Commander Marcus.md"));

        println!("Get linked notes test passed!");
    }

    #[test]
    fn test_get_linked_notes_no_links() {
        let temp_dir = create_example_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Test file with no links
        let request = GetLinkedNotesRequest {
            filename: "Combat Rules.md".to_string(),
        };

        let result = obsidian.get_linked_notes(request).unwrap();

        let content_str = format!("{:?}", result.content[0]);
        assert!(content_str.contains("Combat Rules.md"));
        assert!(content_str.contains("outgoing_links"));
        assert!(content_str.contains("incoming_links"));

        println!("No links test passed!");
    }

    #[test]
    fn test_search_case_sensitivity() {
        let temp_dir = create_example_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Test case sensitive search
        let request = SearchWithContextRequest {
            query: "Crown".to_string(), // Capital C
            context_lines: Some(1),
            regex: Some(false),
            case_sensitive: Some(true),
        };

        let result = obsidian.search_with_context(request).unwrap();

        let content_str = format!("{:?}", result.content[0]);
        // Should find "Crown" with capital C
        assert!(content_str.contains("Crown"));

        println!("Case sensitivity test passed!");
    }

    #[test]
    fn test_bidirectional_links() {
        let temp_dir = create_example_vault();
        let obsidian = Obsidian::new(temp_dir.path().to_path_buf());

        // Test that links work both ways
        let request1 = GetLinkedNotesRequest {
            filename: "campaign_overview.md".to_string(),
        };

        let request2 = GetLinkedNotesRequest {
            filename: "Lord Commander Marcus.md".to_string(),
        };

        let result1 = obsidian.get_linked_notes(request1).unwrap();
        let result2 = obsidian.get_linked_notes(request2).unwrap();

        let content1_str = format!("{:?}", result1.content[0]);
        let content2_str = format!("{:?}", result2.content[0]);

        // Campaign overview should show Marcus file as incoming link
        assert!(content1_str.contains("Lord Commander Marcus.md"));

        // Marcus file should show campaign as outgoing link
        assert!(content2_str.contains("campaign_overview"));

        println!("Bidirectional links test passed!");
    }
}
