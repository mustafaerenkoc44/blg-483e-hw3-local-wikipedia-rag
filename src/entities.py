"""Curated lists of people and places to ingest from Wikipedia.

The required set from the homework brief is included verbatim. Additional
entities give the retriever a wider corpus to exercise its routing logic
(e.g. Hagia Sophia and Topkapi Palace cover the "place in Turkey" question).

Each entry stores:
    name           — display name shown to the user
    wiki_title     — exact Wikipedia article title (case-sensitive)
    aliases        — alternative spellings the query router will match against
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Entity:
    name: str
    wiki_title: str
    type: str
    aliases: tuple[str, ...] = field(default_factory=tuple)


PEOPLE: tuple[Entity, ...] = (
    Entity("Albert Einstein", "Albert Einstein", "person", ("einstein",)),
    Entity("Marie Curie", "Marie Curie", "person", ("curie", "marie sklodowska")),
    Entity("Leonardo da Vinci", "Leonardo da Vinci", "person", ("da vinci", "leonardo")),
    Entity("William Shakespeare", "William Shakespeare", "person", ("shakespeare",)),
    Entity("Ada Lovelace", "Ada Lovelace", "person", ("lovelace", "ada byron")),
    Entity("Nikola Tesla", "Nikola Tesla", "person", ("tesla",)),
    Entity("Lionel Messi", "Lionel Messi", "person", ("messi", "leo messi")),
    Entity("Cristiano Ronaldo", "Cristiano Ronaldo", "person", ("ronaldo", "cr7")),
    Entity("Taylor Swift", "Taylor Swift", "person", ("swift",)),
    Entity("Frida Kahlo", "Frida Kahlo", "person", ("kahlo",)),
    Entity("Mahatma Gandhi", "Mahatma Gandhi", "person", ("gandhi",)),
    Entity("Mustafa Kemal Atatürk", "Mustafa Kemal Atatürk", "person", ("ataturk", "atatürk", "mustafa kemal")),
    Entity("Wolfgang Amadeus Mozart", "Wolfgang Amadeus Mozart", "person", ("mozart",)),
    Entity("Ludwig van Beethoven", "Ludwig van Beethoven", "person", ("beethoven",)),
    Entity("Pablo Picasso", "Pablo Picasso", "person", ("picasso",)),
    Entity("Vincent van Gogh", "Vincent van Gogh", "person", ("van gogh", "vangogh")),
    Entity("Stephen Hawking", "Stephen Hawking", "person", ("hawking",)),
    Entity("Steve Jobs", "Steve Jobs", "person", ("jobs",)),
    Entity("Isaac Newton", "Isaac Newton", "person", ("newton",)),
    Entity("Charles Darwin", "Charles Darwin", "person", ("darwin",)),
)

PLACES: tuple[Entity, ...] = (
    Entity("Eiffel Tower", "Eiffel Tower", "place", ("eiffel",)),
    Entity("Great Wall of China", "Great Wall of China", "place", ("great wall",)),
    Entity("Taj Mahal", "Taj Mahal", "place", ("taj",)),
    Entity("Grand Canyon", "Grand Canyon", "place", ("grand canyon",)),
    Entity("Machu Picchu", "Machu Picchu", "place", ("machu picchu",)),
    Entity("Colosseum", "Colosseum", "place", ("colosseum", "coliseum", "roman colosseum")),
    Entity("Hagia Sophia", "Hagia Sophia", "place", ("hagia sophia", "ayasofya")),
    Entity("Statue of Liberty", "Statue of Liberty", "place", ("statue of liberty", "liberty statue")),
    Entity("Giza pyramid complex", "Giza pyramid complex", "place", ("pyramids of giza", "great pyramid", "egyptian pyramids")),
    Entity("Mount Everest", "Mount Everest", "place", ("everest",)),
    Entity("Stonehenge", "Stonehenge", "place", ("stonehenge",)),
    Entity("Petra", "Petra", "place", ("petra",)),
    Entity("Acropolis of Athens", "Acropolis of Athens", "place", ("acropolis", "parthenon")),
    Entity("Angkor Wat", "Angkor Wat", "place", ("angkor",)),
    Entity("Christ the Redeemer (statue)", "Christ the Redeemer (statue)", "place", ("christ the redeemer", "rio statue")),
    Entity("Sydney Opera House", "Sydney Opera House", "place", ("opera house",)),
    Entity("Burj Khalifa", "Burj Khalifa", "place", ("burj khalifa",)),
    Entity("Niagara Falls", "Niagara Falls", "place", ("niagara",)),
    Entity("Chichen Itza", "Chichen Itza", "place", ("chichen itza", "chichén itzá")),
    Entity("Topkapı Palace", "Topkapı Palace", "place", ("topkapi", "topkapı")),
)

ALL_ENTITIES: tuple[Entity, ...] = PEOPLE + PLACES


def by_name(name: str) -> Entity | None:
    """Look up an entity by display name (case-insensitive)."""
    target = name.strip().lower()
    for ent in ALL_ENTITIES:
        if ent.name.lower() == target:
            return ent
    return None
