"""Generate documentation with example sentences."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import yaml

ORGANIZING_URL_BASE = "https://www.home-assistant.io/docs/organizing"
INTEGRATION_URL_BASE = "https://www.home-assistant.io/integrations"

_DIR = Path(__file__).parent
_PROGRAM_DIR = _DIR.parent
_DOCS_DIR = _PROGRAM_DIR / "docs"
_TESTS_DIR = _PROGRAM_DIR / "tests"
_EXAMPLE_SENTENCES_DIR = _TESTS_DIR / "example_sentences"

INTENT_SLOTS: Dict[str, List[Tuple[str, Tuple[Union[str, Tuple[str, ...]], ...]]]] = {
    "Date and Time": [
        ("HassGetCurrentTime", ("default",)),
        ("HassGetCurrentDate", ("default",)),
    ],
    "Weather and Temperature": [
        ("HassGetWeather", ("default", "name_only")),
        ("HassClimateGetTemperature", ("default", "name_only")),
    ],
    "Lights": [
        (
            "HassTurnOn",
            (
                "domain_light",
                ("name_only", "light"),
                "area_domain_light",
                ("name_area", "light"),
            ),
        ),
        (
            "HassTurnOff",
            (
                "domain_light",
                ("name_only", "light"),
                "area_domain_light",
                ("name_area", "light"),
            ),
        ),
        (
            "HassLightSet",
            (
                "brightness_only",
                "name_brightness",
                "area_brightness",
                "color_only",
                "name_color",
                "area_color",
            ),
        ),
    ],
    # TODO: switches, etc.
    "Sensors": [
        ("HassGetState", (("name_sensor", "sensor"), ("name_sensor", "binary_sensor")))
    ],
    "Doors and Windows": [
        ("HassTurnOn", ("name_cover", "name_area_cover", "area_device_class_cover")),
        ("HassTurnOff", ("name_cover", "name_area_cover", "area_device_class_cover")),
        ("HassGetState", ("name_cover",)),
    ],
    "Locks": [
        ("HassTurnOn", ("name_lock",)),
        ("HassTurnOff", ("name_lock",)),
        ("HassGetState", ("name_lock",)),
    ],
    "Media": [
        ("HassTurnOn", (("name_only", "media_player"),)),
        ("HassTurnOff", (("name_only", "media_player"),)),
        ("HassMediaPause", ("default", ("name_only", "media_player"))),
        ("HassMediaUnpause", ("default", ("name_only", "media_player"))),
        ("HassMediaNext", ("default", ("name_only", "media_player"))),
    ],
    "Timers": [
        (
            "HassStartTimer",
            ("hours_only", "minutes_only", "seconds_only", "hours_minutes"),
        ),
        ("HassPauseTimer", ("default",)),
        ("HassUnpauseTimer", ("default",)),
        ("HassTimerStatus", ("default",)),
    ],
    "Scenes and Scripts": [("HassTurnOn", ("name_scene", "name_script"))],
    # "Todo": [("HassListAddItem", ("name_item",))],
    "Miscellaneous": [("HassNevermind", ("default",))],
}


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True)
    args = parser.parse_args()

    lang_code = args.language

    with open(_PROGRAM_DIR / "intents.yaml", "r", encoding="utf-8") as intents_file:
        intents = yaml.safe_load(intents_file)

    example_sentences_path = _EXAMPLE_SENTENCES_DIR / f"{lang_code}.json"
    with open(example_sentences_path, "r", encoding="utf-8") as example_sentences_file:
        example_sentences_dict = json.load(example_sentences_file)

    # Sort by longest sentence first
    example_sentences_dict = {
        k: example_sentences_dict[k] for k in sorted(example_sentences_dict, key=len)
    }

    # Markdown
    print(f"# {lang_code}")
    print("")

    used_entity_domains: Set[str] = set()
    for title, intent_slots in INTENT_SLOTS.items():
        # Avoid printing title unless there are some examples
        title_printed = False

        for intent_name, slot_combos in intent_slots:
            intent_info = intents[intent_name]

            for slot_combo_name in slot_combos:
                combo_domain: Optional[str] = None
                if not isinstance(slot_combo_name, str):
                    # (combo_name, combo_domain)
                    slot_combo_name, combo_domain = slot_combo_name

                slot_combo_info = intent_info["slot_combinations"][slot_combo_name]
                if not combo_domain:
                    combo_domain = slot_combo_info.get("domain")

                for example_text, example_info in example_sentences_dict.items():
                    if example_info["intent"] != intent_name:
                        continue

                    if example_info["slot_combination"] != slot_combo_name:
                        continue

                    if combo_domain and (example_info.get("domain") != combo_domain):
                        continue

                    if not title_printed:
                        print("##", title)
                        print("")
                        title_printed = True

                    print("-", f'"{example_text}"')

                    # Context area
                    if example_info.get("context_area"):
                        print("    - Requires voice satellite to be in an [area][]")

                    # name/area/floor
                    example_slots = example_info.get("slots")
                    if example_slots:
                        entity_name = example_slots.get("name")
                        if entity_name:
                            assert (
                                combo_domain
                            ), f"Missing domain for '{intent_name}' slot combination '{slot_combo_name}'"
                            assert isinstance(combo_domain, str)

                            print(
                                f'    - Requires a [{combo_domain}][] named "{entity_name}" to be exposed'
                            )
                            used_entity_domains.add(combo_domain)

                        area_name = example_slots.get("area")
                        if area_name:
                            print(f'    - Requires an [area][] named "{area_name}"')

                    # Only print one example of each slot combination
                    break

        print("")

    # Links
    print("<-- Links -->")
    for org_type in ("area", "floor"):
        print(f"[{org_type}]:", f"{ORGANIZING_URL_BASE}/#{org_type}")

    for domain in used_entity_domains:
        print(f"[{domain}]:", f"{INTEGRATION_URL_BASE}/{domain}")

    print("")


if __name__ == "__main__":
    main()
