"""The agent's default name is Captain; an untouched old default reads as it."""

from captain_claw.personality import (
    DEFAULT_PERSONALITY,
    Personality,
    parse_personality_markdown,
    personality_to_markdown,
    personality_to_prompt_block,
)

OLD_DEFAULT = Personality(
    name="Captain Claw",
    description="A powerful AI assistant with claws — part of the Captain Claw family.",
)


def test_the_default_is_captain():
    assert DEFAULT_PERSONALITY.name == "Captain"
    assert "Captain Claw" not in DEFAULT_PERSONALITY.description
    assert personality_to_prompt_block(DEFAULT_PERSONALITY).startswith("You are Captain.\n")


def test_an_untouched_old_default_on_disk_reads_as_captain():
    p = parse_personality_markdown(personality_to_markdown(OLD_DEFAULT))
    assert (p.name, p.description) == (DEFAULT_PERSONALITY.name, DEFAULT_PERSONALITY.description)
    assert p.background == OLD_DEFAULT.background


def test_a_name_or_description_someone_changed_stays():
    kept_name = Personality(name="Captain Claw", description="Our research desk.")
    p = parse_personality_markdown(personality_to_markdown(kept_name))
    assert (p.name, p.description) == ("Captain Claw", "Our research desk.")

    renamed = Personality(name="Iris", description=OLD_DEFAULT.description)
    p = parse_personality_markdown(personality_to_markdown(renamed))
    assert (p.name, p.description) == ("Iris", OLD_DEFAULT.description)


def test_the_family_suffix_says_captain():
    block = personality_to_prompt_block(Personality(name="Iris"))
    assert block.startswith("You are Iris of the Captain family.\n")
    assert "Captain Claw" not in block
