#!/usr/bin/env python3
"""
VehicleVision — Static Automotive & Statutory Constants
=========================================================
Centralized reference dictionaries, statutory Parivahan RC field definitions,
Indian state/UT RTO codes, OEM manufacturers, and model-to-body mappings.

Imported by getText.py, detectVehicle.py, server.py, etc.
"""

from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 1. 27 Canonical Statutory Registration Certificate (RC) Fields
# ---------------------------------------------------------------------------
RC_CANONICAL_FIELDS: List[str] = [
    "registration_number",
    "registered_owner",
    "sdw_of",
    "purpose",
    "registration_date",
    "colour",
    "fuel",
    "vehicle_class",
    "body_type",
    "manufacturer",
    "chassis_number",
    "engine_number",
    "model_number",
    "manufacturing_date",
    "tax_paid_upto",
    "registration_validity",
    "hypothecated_to",
    "seat_capacity",
    "standing_capacity",
    "number_of_cylinders",
    "owner_serial",
    "unladen_weight",
    "cubic_capacity",
    "wheel_base",
    "rlw",
    "address",
    "issuing_authority",
]

# ---------------------------------------------------------------------------
# 2. Indian State and Union Territory RTO Prefixes
# ---------------------------------------------------------------------------
STATE_CODES: List[str] = [
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN", "GA", "GJ",
    "HR", "HP", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB", "OR"
]

# ---------------------------------------------------------------------------
# 3. Major Indian Automotive OEM Manufacturers
# ---------------------------------------------------------------------------
INDIAN_OEM_MAKES: List[str] = [
    "MARUTI SUZUKI", "MARUTI", "SUZUKI",
    "HYUNDAI", "TATA MOTORS", "TATA",
    "MAHINDRA & MAHINDRA", "MAHINDRA",
    "HONDA", "TOYOTA", "KIA",
    "VOLKSWAGEN", "VW", "SKODA",
    "MG MOTOR", "MG", "RENAULT", "NISSAN", "FORD",
    "JEEP", "BMW", "MERCEDES-BENZ", "MERCEDES", "BENZ", "AUDI",
    "VOLVO", "JAGUAR", "LAND ROVER", "FORCE MOTORS", "FORCE",
    "HERO", "ROYAL ENFIELD", "BAJAJ", "TVS", "YAMAHA", "ASHOK LEYLAND"
]

# ---------------------------------------------------------------------------
# 4. Canonical Vehicle Model to (OEM Make, Body Type) Dictionary
# ---------------------------------------------------------------------------
INDIAN_MODEL_TO_MAKE_BODY: Dict[str, Tuple[str, str]] = {
    # Kia
    "SONET": ("Kia", "SUV"),
    "SELTOS": ("Kia", "SUV"),
    "CARENS": ("Kia", "MUV"),
    "CARNIVAL": ("Kia", "MUV"),
    "EV6": ("Kia", "SUV"),
    # Hyundai
    "CRETA": ("Hyundai", "SUV"),
    "VENUE": ("Hyundai", "SUV"),
    "EXTER": ("Hyundai", "SUV"),
    "ALCAZAR": ("Hyundai", "SUV"),
    "TUCSON": ("Hyundai", "SUV"),
    "VERNA": ("Hyundai", "Sedan"),
    "AURA": ("Hyundai", "Sedan"),
    "I20": ("Hyundai", "Hatchback"),
    "GRAND I10": ("Hyundai", "Hatchback"),
    "I10": ("Hyundai", "Hatchback"),
    "SANTRO": ("Hyundai", "Hatchback"),
    "KONA": ("Hyundai", "SUV"),
    "IONIQ 5": ("Hyundai", "SUV"),
    # Maruti Suzuki
    "BREZZA": ("Maruti Suzuki", "SUV"),
    "VITARA BREZZA": ("Maruti Suzuki", "SUV"),
    "GRAND VITARA": ("Maruti Suzuki", "SUV"),
    "FRONX": ("Maruti Suzuki", "SUV"),
    "JIMNY": ("Maruti Suzuki", "SUV"),
    "ERTIGA": ("Maruti Suzuki", "MUV"),
    "XL6": ("Maruti Suzuki", "MUV"),
    "INVICTO": ("Maruti Suzuki", "MUV"),
    "DZIRE": ("Maruti Suzuki", "Sedan"),
    "CIAZ": ("Maruti Suzuki", "Sedan"),
    "SWIFT": ("Maruti Suzuki", "Hatchback"),
    "BALENO": ("Maruti Suzuki", "Hatchback"),
    "ALTO": ("Maruti Suzuki", "Hatchback"),
    "ALTO K10": ("Maruti Suzuki", "Hatchback"),
    "WAGONR": ("Maruti Suzuki", "Hatchback"),
    "WAGON R": ("Maruti Suzuki", "Hatchback"),
    "CELERIO": ("Maruti Suzuki", "Hatchback"),
    "IGNIS": ("Maruti Suzuki", "Hatchback"),
    "S-PRESSO": ("Maruti Suzuki", "Hatchback"),
    "EECO": ("Maruti Suzuki", "Van"),
    "RITZ": ("Maruti Suzuki", "Hatchback"),
    "SX4": ("Maruti Suzuki", "Sedan"),
    "S-CROSS": ("Maruti Suzuki", "Crossover"),
    "SCROSS": ("Maruti Suzuki", "Crossover"),
    # Tata
    "NEXON": ("Tata", "SUV"),
    "PUNCH": ("Tata", "SUV"),
    "HARRIER": ("Tata", "SUV"),
    "SAFARI": ("Tata", "SUV"),
    "CURVV": ("Tata", "SUV"),
    "TIGOR": ("Tata", "Sedan"),
    "TIAGO": ("Tata", "Hatchback"),
    "ALTROZ": ("Tata", "Hatchback"),
    "INDICA": ("Tata", "Hatchback"),
    "INDIGO": ("Tata", "Sedan"),
    "HEXA": ("Tata", "MUV"),
    # Mahindra
    "SCORPIO": ("Mahindra", "SUV"),
    "SCORPIO-N": ("Mahindra", "SUV"),
    "THAR": ("Mahindra", "SUV"),
    "THAR ROXX": ("Mahindra", "SUV"),
    "XUV700": ("Mahindra", "SUV"),
    "XUV300": ("Mahindra", "SUV"),
    "XUV3XO": ("Mahindra", "SUV"),
    "XUV400": ("Mahindra", "SUV"),
    "BOLERO": ("Mahindra", "SUV"),
    "BOLERO NEO": ("Mahindra", "SUV"),
    "MARAZZO": ("Mahindra", "MUV"),
    "KUV100": ("Mahindra", "SUV"),
    "TUV300": ("Mahindra", "SUV"),
    # Toyota
    "FORTUNER": ("Toyota", "SUV"),
    "HYRYDER": ("Toyota", "SUV"),
    "URBAN CRUISER": ("Toyota", "SUV"),
    "INNOVA": ("Toyota", "MUV"),
    "INNOVA CRYSTA": ("Toyota", "MUV"),
    "INNOVA HYCROSS": ("Toyota", "MUV"),
    "CAMRY": ("Toyota", "Sedan"),
    "GLANZA": ("Toyota", "Hatchback"),
    "RUMION": ("Toyota", "MUV"),
    "ETIOS": ("Toyota", "Sedan"),
    # Honda
    "CITY": ("Honda", "Sedan"),
    "AMAZE": ("Honda", "Sedan"),
    "CIVIC": ("Honda", "Sedan"),
    "ELEVATE": ("Honda", "SUV"),
    "WR-V": ("Honda", "SUV"),
    "JAZZ": ("Honda", "Hatchback"),
    # Volkswagen
    "TAIGUN": ("Volkswagen", "SUV"),
    "VIRTUS": ("Volkswagen", "Sedan"),
    "POLO": ("Volkswagen", "Hatchback"),
    "VENTO": ("Volkswagen", "Sedan"),
    "TIGUAN": ("Volkswagen", "SUV"),
    # Skoda
    "KUSHAQ": ("Skoda", "SUV"),
    "SLAVIA": ("Skoda", "Sedan"),
    "KODIAQ": ("Skoda", "SUV"),
    "OCTAVIA": ("Skoda", "Sedan"),
    "SUPERB": ("Skoda", "Sedan"),
    "RAPID": ("Skoda", "Sedan"),
    # MG
    "HECTOR": ("MG", "SUV"),
    "ASTOR": ("MG", "SUV"),
    "ZS EV": ("MG", "SUV"),
    "COMET EV": ("MG", "Hatchback"),
    "GLOSTER": ("MG", "SUV"),
    "WINDSOR": ("MG", "MUV"),
    # Renault & Nissan
    "TRIBER": ("Renault", "MUV"),
    "KIGER": ("Renault", "SUV"),
    "KWID": ("Renault", "Hatchback"),
    "DUSTER": ("Renault", "SUV"),
    "MAGNITE": ("Nissan", "SUV"),
    "KICKS": ("Nissan", "SUV"),
    "SUNNY": ("Nissan", "Sedan"),
    "MICRA": ("Nissan", "Hatchback"),
    # Ford & Jeep
    "ECOSPORT": ("Ford", "SUV"),
    "ENDEAVOUR": ("Ford", "SUV"),
    "FIGO": ("Ford", "Hatchback"),
    "COMPASS": ("Jeep", "SUV"),
    "MERIDIAN": ("Jeep", "SUV"),
    # Luxury Brands
    "3 SERIES": ("BMW", "Sedan"),
    "5 SERIES": ("BMW", "Sedan"),
    "X1": ("BMW", "SUV"),
    "X3": ("BMW", "SUV"),
    "X5": ("BMW", "SUV"),
    "C-CLASS": ("Mercedes-Benz", "Sedan"),
    "E-CLASS": ("Mercedes-Benz", "Sedan"),
    "GLA": ("Mercedes-Benz", "SUV"),
    "GLC": ("Mercedes-Benz", "SUV"),
    "A4": ("Audi", "Sedan"),
    "A6": ("Audi", "Sedan"),
    "Q3": ("Audi", "SUV"),
    "Q5": ("Audi", "SUV")
}

# ---------------------------------------------------------------------------
# 5. Dealer Brands, Keywords & Rear Badges
# ---------------------------------------------------------------------------
DEALER_NAMES: List[str] = [
    "SPINNY", "SPINY", "CARS24", "CARDEKHO", "TRUE VALUE", "OLX"
]

REAR_BADGES: List[str] = [
    "TITANIUM", "TITANUIOM", "TDCI", "TDOI", "ATIKA BAR", "VITARA",
    "BREZZA", "SWIFT", "CRETA", "SELTOS", "ECOSPORT", "VXI", "ZXI",
    "GRAND", "JANGRA", "ZDI", "ZDI+", "ZXI+", "VDI", "LDI", "LXI", "SHVS", "HYBRID",
    "MARUTI", "SUZUKI", "MARUTI SUZUKI", "MARUT", "SUZUK",
    "ALLGRIP", "BOOSTERJET", "4X4", "AWD", "4WD", "DIESEL", "CNG",
    "CRDI", "VTEC", "IVTEC", "IDTEC", "TSI", "TDI", "TURBO", "GDI", "MPI",
    "1.4", "1.6", "1.2", "1.0", "1.5", "2.0",
    "S PLUS", "S+", "SX", "SX(O)", "EX", "E", "S", "SV", "VX", "ZX",
    "DELTA", "ZETA", "ALPHA", "SIGMA",
    "HTE", "HTK", "HTX", "GTX", "GT LINE", "GT-LINE", "X-LINE", "XLINE",
    "TRENDLINE", "COMFORTLINE", "HIGHLINE", "DYNAMIC",
    "PETROL", "SPORTZ", "ASTA", "MAGNA", "ERA",
    "PURE", "ADVENTURE", "ACCOMPLISHED", "CREATIVE", "SMART", "FEARLESS",
    "NEXA", "ARENA", "RANOVITARA", "AMAZE", "BALENO", "VERNA", "CITY",
    "INNOVA", "FORTUNER", "NEXON", "HARRIER", "SAFARI", "PUNCH", "TIAGO", "TIGOR",
    "SCORPIO", "THAR", "XUV700", "XUV300", "BOLERO", "ALTO", "WAGONR",
    "CELERIO", "IGNIS", "DZIRE", "CIAZ", "ERTIGA", "XL6", "CARENS", "SONET"
]


INTERIOR_KEYWORDS: List[str] = [
    "FM91", "AUDIO", "TRIP", "KM/H", "RPM", "AIRBAG", "CLIMATE", "INFOTAINMENT", "BLUETOOTH", "SPEEDOMETER"
]

RC_KEYWORDS: List[str] = [
    "REGN", "CHASSIS", "ENGINE", "SDM", "RTO", "DTO", "OWNER", "VALIDITY", "FUEL",
    "CYLINDERS", "UNLADEN", "CUBIC", "FORM 23", "REGISTRATION"
]

# ---------------------------------------------------------------------------
# 6. Lookup Helper Functions
# ---------------------------------------------------------------------------
def lookup_model_info(model_query: str) -> Optional[Tuple[str, str]]:
    """Looks up a model name in uppercase and returns (Make, BodyType) or None."""
    if not model_query:
        return None
    return INDIAN_MODEL_TO_MAKE_BODY.get(model_query.upper().strip())


def is_valid_state_code(code: str) -> bool:
    """Returns True if the 2-letter state code is a recognized Indian RTO prefix."""
    if not code:
        return False
    return code.upper().strip() in STATE_CODES


# ---------------------------------------------------------------------------
# 7. Primary Vehicle Colors (White, Silver, Black, Blue, Red, Brown only)
# ---------------------------------------------------------------------------
PRIMARY_CAR_COLORS: List[str] = ["White", "Silver", "Black", "Blue", "Red", "Brown"]

COLOR_TO_PRIMARY_MAP: Dict[str, str] = {
    # White
    "WHITE": "White",
    "PEARL WHITE": "White",
    "POLAR WHITE": "White",
    "ARCTIC WHITE": "White",
    "GLACIER WHITE": "White",
    "PRISTINE WHITE": "White",
    "OFF-WHITE": "White",
    "CREAM": "White",
    "IVORY": "White",
    # Silver
    "SILVER": "Silver",
    "SILVER FROST": "Silver",
    "METALLIC SILVER": "Silver",
    "SILVER METALLIC": "Silver",
    "BRIGHT SILVER": "Silver",
    "SILKY SILVER": "Silver",
    "ICED SILVER": "Silver",
    "STARLIGHT SILVER": "Silver",
    "PREMIUM SILVER": "Silver",
    "ARCTIC SILVER": "Silver",
    "LIGHT GREY": "Silver",
    "LIGHT GRAY": "Silver",
    # Black
    "BLACK": "Black",
    "MIDNIGHT BLACK": "Black",
    "PHANTOM BLACK": "Black",
    "GREY": "Black",
    "GRAY": "Black",
    "DAYTONA GREY": "Black",
    "MAGMA GREY": "Black",
    "GRANITE GREY": "Black",
    "CHARCOAL": "Black",
    "DARK GRAY": "Black",
    # Blue
    "BLUE": "Blue",
    "ROYALE BLUE": "Blue",
    "NAVY BLUE": "Blue",
    "SKY BLUE": "Blue",
    "TEAL": "Blue",
    "CYAN": "Blue",
    # Red
    "RED": "Red",
    "FLAME RED": "Red",
    "FIRE RED": "Red",
    "CHERRY RED": "Red",
    "MAROON": "Red",
    "BURGUNDY": "Red",
    "CRIMSON": "Red",
    # Brown
    "BROWN": "Brown",
    "DESERT GOLD": "Brown",
    "GOLD": "Brown",
    "BEIGE": "Brown",
    "BRONZE": "Brown",
    "COPPER": "Brown",
    "ORANGE": "Brown",
    "YELLOW": "Brown",
    "TAN": "Brown",
}


def normalize_car_color(color_name: Optional[str]) -> Optional[str]:
    """Normalizes any vehicle color string strictly to one of: White, Silver, Black, Blue, Red, Brown."""
    if not color_name:
        return None
    raw = str(color_name).strip().upper()
    if raw in COLOR_TO_PRIMARY_MAP:
        return COLOR_TO_PRIMARY_MAP[raw]
    for key, val in COLOR_TO_PRIMARY_MAP.items():
        if key in raw:
            return val
    for p in ["WHITE", "SILVER", "BLACK", "BLUE", "RED", "BROWN"]:
        if p in raw:
            return p.title()
    return None

