"""Shared fixtures: small frames and known-good identifiers."""
import numpy as np
import pandas as pd
import pytest

# Aadhaar numbers whose last digit is the correct Verhoeff check digit.
VALID_AADHAAR = ["234567890124", "491039587666", "876543210988", "212121212120"]
# Same numbers with a wrong check digit, plus shapes Aadhaar forbids.
INVALID_AADHAAR = ["234567890123", "491039587660", "012345678901", "112345678901", "23456789012"]

# Luhn-valid card numbers (the published test numbers).
VALID_CARDS = ["4111111111111111", "5555555555554444", "378282246310005", "6011000990139424"]
INVALID_CARDS = ["4111111111111112", "5555555555554443", "1234567890123456", "411111111111"]

VALID_PANS = ["ABCPD1234E", "XYZPK9876F", "AAACT1234Z", "BNZAA2318J"]
INVALID_PANS = ["ABCXD1234E", "ABCP1234E", "ABCPD12345", "12CPD1234E"]


@pytest.fixture
def customers():
    """A small customer table with one column per detector."""
    return pd.DataFrame(
        {
            "full_name": ["Asha Rao", "Vikram Nair", "Meera Iyer"],
            "email": ["asha@example.com", "vikram@example.org", "meera@example.net"],
            "phone": ["9876543210", "+91 91234 56780", "98765-43210"],
            "aadhaar": ["2345 6789 0124", "4910 3958 7666", "8765 4321 0988"],
            "pan": VALID_PANS[:3],
            "card_number": VALID_CARDS[:3],
            "ip_address": ["192.168.1.10", "8.8.8.8", "10.0.0.255"],
            "website": ["https://example.com", "www.example.org", "http://example.net/x"],
            "dob": ["1990-05-14", "14/02/1985", "3 March 1977"],
            "pincode": ["560001", "110011", "400051"],
            "address": ["12 MG Road, Bengaluru", "Flat 4B, 22 Park Street", "742 Evergreen Terrace"],
            "gender": ["F", "M", "F"],
            "order_amount": [1299, 45999, 320],
            "order_id": [100234, 100235, 100236],
        }
    )


@pytest.fixture
def clean_frame():
    """A frame with nothing personal in it."""
    return pd.DataFrame(
        {
            "sku": ["A-1", "A-2", "A-3"],
            "units": [3, 9, 1],
            "price": [19.5, 4.25, 99.0],
            "warehouse": ["north", "south", "north"],
        }
    )


@pytest.fixture
def big_frame():
    """50,000 rows: the performance bar from the spec."""
    n = 50_000
    rng = np.random.RandomState(0)
    return pd.DataFrame(
        {
            "email": [f"user{i}@example.com" for i in range(n)],
            "mobile": [f"9{i:09d}"[:10] for i in range(n)],
            "amount": rng.randint(1, 100_000, n),
            "note": ["shipped from the warehouse"] * n,
        }
    )
