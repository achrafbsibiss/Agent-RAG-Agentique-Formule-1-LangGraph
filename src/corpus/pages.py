"""Plan du corpus documentaire Formule 1.

Chaque entree associe un titre de page Wikipedia a une categorie metier. La
categorie devient une metadonnee du chunk indexe : c'est elle qui permet a
l'outil de recherche filtree de restreindre l'espace de recherche.

Le corpus est volontairement etendu a plusieurs axes (reglement, ecuries,
pilotes, circuits, technique, saisons) afin que les questions complexes
exigent de croiser plusieurs documents.
"""

from __future__ import annotations

# (titre Wikipedia, categorie, langue)
PAGES: list[tuple[str, str, str]] = [
    # ---------------- Reglement sportif et technique ----------------
    ("Formula One regulations", "reglement", "en"),
    ("Racing flags", "reglement", "en"),
    ("Safety car", "reglement", "en"),
    ("Parc fermé", "reglement", "en"),
    ("Formula One racing", "reglement", "en"),
    ("List of Formula One points systems", "reglement", "en"),
    ("Safety in Formula One", "reglement", "en"),
    ("Halo (safety device)", "reglement", "en"),

    # ---------------- Ecuries ----------------
    ("Mercedes-Benz in Formula One", "ecurie", "en"),
    ("Scuderia Ferrari", "ecurie", "en"),
    ("Red Bull Racing", "ecurie", "en"),
    ("McLaren", "ecurie", "en"),
    ("Alpine F1 Team", "ecurie", "en"),
    ("Aston Martin in Formula One", "ecurie", "en"),
    ("Williams Racing", "ecurie", "en"),
    ("RB Formula One Team", "ecurie", "en"),
    ("Sauber Motorsport", "ecurie", "en"),
    ("Haas F1 Team", "ecurie", "en"),

    # ---------------- Pilotes ----------------
    ("Max Verstappen", "pilote", "en"),
    ("Lewis Hamilton", "pilote", "en"),
    ("Charles Leclerc", "pilote", "en"),
    ("Lando Norris", "pilote", "en"),
    ("Fernando Alonso", "pilote", "en"),
    ("Sergio Pérez", "pilote", "en"),
    ("Carlos Sainz Jr.", "pilote", "en"),
    ("George Russell", "pilote", "en"),
    ("Oscar Piastri", "pilote", "en"),
    ("Ayrton Senna", "pilote", "en"),
    ("Michael Schumacher", "pilote", "en"),
    ("Sebastian Vettel", "pilote", "en"),

    # ---------------- Circuits ----------------
    ("Circuit de Monaco", "circuit", "en"),
    ("Silverstone Circuit", "circuit", "en"),
    ("Monza Circuit", "circuit", "en"),
    ("Circuit de Spa-Francorchamps", "circuit", "en"),
    ("Suzuka International Racing Course", "circuit", "en"),
    ("Circuit of the Americas", "circuit", "en"),
    ("Marina Bay Street Circuit", "circuit", "en"),
    ("Baku City Circuit", "circuit", "en"),

    # ---------------- Glossaire / vocabulaire ----------------
    ("Glossary of motorsport terms", "glossaire", "en"),
    ("Drag reduction system", "glossaire", "en"),
    ("Formula One tyres", "glossaire", "en"),

    # ---------------- Technique et motorisation ----------------
    ("Formula One car", "technique", "en"),
    ("Formula One engines", "technique", "en"),
    ("Kinetic energy recovery system", "technique", "en"),
    ("Ground effect (cars)", "technique", "en"),
    ("Downforce", "technique", "en"),
    # Approfondissement moteur : architecture, suralimentation, injection.
    ("Turbocharger", "technique", "en"),
    ("V6 engine", "technique", "en"),
    ("Fuel injection", "technique", "en"),

    # ---------------- Strategie de course ----------------
    ("Formula One race weekend", "strategie", "en"),
    ("Racing setup", "strategie", "en"),
    ("Overtaking", "strategie", "en"),
    ("Drafting (aerodynamics)", "strategie", "en"),
    ("Pit stop", "strategie", "en"),

    # ---------------- Saisons 2005-2025 ----------------
    ("2005 Formula One World Championship", "saison", "en"),
    ("2006 Formula One World Championship", "saison", "en"),
    ("2007 Formula One World Championship", "saison", "en"),
    ("2008 Formula One World Championship", "saison", "en"),
    ("2009 Formula One World Championship", "saison", "en"),
    ("2010 Formula One World Championship", "saison", "en"),
    ("2011 Formula One World Championship", "saison", "en"),
    ("2012 Formula One World Championship", "saison", "en"),
    ("2013 Formula One World Championship", "saison", "en"),
    ("2014 Formula One World Championship", "saison", "en"),
    ("2015 Formula One World Championship", "saison", "en"),
    ("2016 Formula One World Championship", "saison", "en"),
    ("2017 Formula One World Championship", "saison", "en"),
    ("2018 Formula One World Championship", "saison", "en"),
    ("2019 Formula One World Championship", "saison", "en"),
    ("2020 Formula One World Championship", "saison", "en"),
    ("2021 Formula One World Championship", "saison", "en"),
    ("2022 Formula One World Championship", "saison", "en"),
    ("2023 Formula One World Championship", "saison", "en"),
    ("2024 Formula One World Championship", "saison", "en"),
    ("2025 Formula One World Championship", "saison", "en"),

    # ---------------- Histoire et palmares ----------------
    ("Formula One", "histoire", "en"),
    ("History of Formula One", "histoire", "en"),
    ("List of Formula One World Drivers' Champions", "histoire", "en"),
    ("List of Formula One World Constructors' Champions", "histoire", "en"),

    # ---------------- Sources francophones ----------------
    # Ancrent le corpus dans le vocabulaire FR (ecurie, pneus tendres, etc.),
    # ce qui ameliore le rappel sur les questions posees en francais.
    ("Formule 1", "histoire", "fr"),
    ("Championnat du monde de Formule 1 2024", "saison", "fr"),
    ("Grand Prix automobile de Monaco", "circuit", "fr"),
    ("Système de réduction de la traînée", "glossaire", "fr"),
]
