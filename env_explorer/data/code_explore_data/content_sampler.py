import random
from faker import Faker
from typing import List
from unidecode import unidecode
import re
import pandas as pd
from pathlib import Path


encoding_to_locale = {
    'utf-8': ['en_US', 'en_GB'],
    'latin-1': ['fr_FR', 'de_DE', 'es_ES'],
    'utf-16': ['zh_CN', 'ja_JP', 'ko_KR'],
}
encoding_to_onename = {
    'utf-8': 'Alex Doe',
    'latin-1': 'Élodie Dubois',
    'utf-16': '小明',
}

id_keys = ['name', 'username']
number_keys = ['salary', 'age', 'price', 'rating', 'score', 'total_count']
text_keys = ['note', 'description', 'comment', 'remark', 'summary', 'message', 'review']

def translit(name):
    try:
        # transliterate & clean
        translit = unidecode(name)
        translit = re.sub(r'[^A-Za-z0-9]+', ' ', translit).strip().lower()
        base_parts = translit.split()
        if len(base_parts) >= 2:
            base = base_parts[0] + "_" + base_parts[-1]
        else:
            base = base_parts[0] if base_parts else "user"
    except Exception:
        base = "user"
    return base

class CSVContentSampler:
    """Samples content for CSV files using Faker."""

    def __init__(
            self, 
            delimiter: str = ',', 
            quotechar: str = '"', 
            encoding: str = 'utf-8', 
            path: str = 'sample.csv'):
        self.delimiter = delimiter
        self.quotechar = quotechar
        self.encoding = encoding
        self.locale = random.choice(encoding_to_locale.get(encoding, ['en_US']))
        self.faker = Faker(self.locale)
        self.safe_name = random.choice(encoding_to_onename.get(encoding, ['Alex Doe']))
        # sample keys to include
        self.id_keys = random.sample(id_keys, k=random.randint(1, len(id_keys)))
        self.number_keys = random.sample(number_keys, k=random.randint(1, 3))
        self.text_keys = random.sample(text_keys, k=random.randint(1, 2))
        self.path = path
    
    def _sample_name_username(self):
        """Generate (name, username) pair."""
        name = self.faker.name()
        base_username = translit(name)
        suffix_num = self.faker.bothify("####")       # → '8392'

        return {"name": name, "username": f"{base_username}_{suffix_num}"}
    def _sample_number(self):
        """Return a random positive integer."""
        return random.randint(1, 100)
    def _sample_text(self):
        """Generate realistic CSV text with embedded quotes and commas."""
        # 4 short fragments
        sents = [
            self.faker.sentence(nb_words=random.randint(2, 5)).strip().rstrip(".")
            for _ in range(4)
        ]
        # remove any internal quotes
        # sents = [re.sub(r'["\']', '', s) for s in sents]
        q = self.quotechar
        d = self.delimiter
        # 2. ESCAPING LOGIC: If the quotechar appears in the text, 
        # standard CSV format requires it to be doubled (e.g. ' becomes '')
        def escape(t):
            return t.replace(q, q + q)

        formats = [
            f"{sents[0]}{d} {sents[1]} {sents[2]}",
            f"{sents[0]}: {sents[1]}{d} {sents[2]}",
            f"Note: {sents[0]}{d} then {sents[1]}",
            f"({sents[0]}) {d} {sents[1]} and {sents[2]}",
            f"{sents[0]}; {sents[1]} {d} {sents[2]}",
        ]
        raw_text = random.choice(formats)

        return f"{q}{escape(raw_text)}{q}"


    def sample_records(self):
        records = []
        n_rows = random.randint(20, 50)
        for _ in range(n_rows):
            record = {}

            # ID features
            id_dict = self._sample_name_username()
            for key in self.id_keys:
                record[key] = id_dict.get(key, self.faker.name())

            # Textual features
            for key in self.text_keys:
                record[key] = self._sample_text()

            # Numeric features
            for key in self.number_keys:
                record[key] = self._sample_number()

            records.append(record)

        # Replace one random record's name with the "special" encoding test name
        if "name" in self.id_keys and records:
            records[random.randint(0, len(records) - 1)]["name"] = self.safe_name
        
        # For each numeric key, ensure at least one is None
        for key in self.number_keys:
            if records:
                records[random.randint(0, len(records) - 1)][key] = None
                

        return records

    def write_csv(self):
        records = self.sample_records()
        df = pd.DataFrame(records)

        # Write with specified encoding, delimiter, and quotechar
        try:
            df.to_csv(
                self.path,
                encoding=self.encoding,
                sep=self.delimiter,
                quotechar=self.quotechar,
                index=False,
                header=True,
            )
        except Exception as e:
            # df.head()
            print(f"Sampled DataFrame before error:")
            print(df.head())
            print("formatting params:", self.encoding, self.delimiter, self.quotechar)
            print(f"Error writing CSV with encoding {self.encoding}: {e}")
            raise e
        
        # df.to_csv(
        #     self.path,
        #     encoding=self.encoding,
        #     sep=self.delimiter,
        #     quotechar=self.quotechar,
        #     index=False,
        #     header=True,
        # )
        # print(f"✅ CSV written: {self.path} ({self.encoding}, {self.delimiter=}, {self.quotechar=})")
        # print(f"Columns: {list(df.columns)}")

        code_to_read = (
            f"import pandas as pd\n"
            f"df = pd.read_csv(\n"
            f"    \"{{path}}\",\n"
            f"    encoding={self.encoding!r},\n"
            f"    sep={self.delimiter!r},\n"
            f"    quotechar={self.quotechar!r},\n"
            f"    header=0\n"
            f")"
        )
        meta_info = {
            "encoding": self.encoding,
            "delimiter": self.delimiter,
            "quotechar": self.quotechar,
            "path": Path(self.path).name,
            "n_rows": len(df),
            "locale": self.locale,
            "key_names": {
                "id_keys": self.id_keys,
                "number_keys": self.number_keys,
                "text_keys": self.text_keys,
            },
            "read_code": code_to_read,
        }
        # print(f"✅ CSV written: {self.path}")
        # print(f"Columns: {list(df.columns)}")
        # print("\nTo read this file correctly, run:\n")
        # print(code_to_read)
        return df, meta_info

class CSVContentSampler_independent:
    """Samples content for CSV files using Faker."""

    def __init__(
            self, 
            delimiter: str = ',', 
            quotechar: str = '"', 
            skiprows: int = 0, 
            path: str = 'sample.csv'):
        self.delimiter = delimiter
        self.quotechar = quotechar
        self.skiprows = skiprows
        self.encoding = 'utf-8'
        self.locale = random.choice(encoding_to_locale.get(self.encoding, ['en_US']))
        self.faker = Faker(self.locale)
        self.safe_name = random.choice(encoding_to_onename.get(self.encoding, ['Alex Doe']))
        # sample keys to include
        self.id_keys = random.sample(id_keys, k=random.randint(1, len(id_keys)))
        self.number_keys = random.sample(number_keys, k=random.randint(1, 3))
        self.text_keys = random.sample(text_keys, k=random.randint(1, 2))
        self.path = path
    
    def _sample_name_username(self):
        """Generate (name, username) pair."""
        name = self.faker.name()
        base_username = translit(name)
        suffix_num = self.faker.bothify("####")       # → '8392'

        return {"name": name, "username": f"{base_username}_{suffix_num}"}
    def _sample_number(self, offset=0):
        """Return a random positive integer."""
        base = random.randint(1, 100)
        return base + offset
    def _sample_text(self):
        """Generate realistic CSV text with embedded quotes and commas."""
        # 4 short fragments
        sents = [
            self.faker.sentence(nb_words=random.randint(2, 5)).strip().rstrip(".")
            for _ in range(4)
        ]
        # remove any internal quotes
        # sents = [re.sub(r'["\']', '', s) for s in sents]
        q = self.quotechar
        d = self.delimiter
        # 2. ESCAPING LOGIC: If the quotechar appears in the text, 
        # standard CSV format requires it to be doubled (e.g. ' becomes '')
        def escape(t):
            return t.replace(q, q + q)

        formats = [
            f"{sents[0]}{d} {sents[1]} {sents[2]}",
            f"{sents[0]}: {sents[1]}{d} {sents[2]}",
            f"Note: {sents[0]}{d} then {sents[1]}",
            f"({sents[0]}) {d} {sents[1]} and {sents[2]}",
            f"{sents[0]}; {sents[1]} {d} {sents[2]}",
        ]
        raw_text = random.choice(formats)

        return f"{q}{escape(raw_text)}{q}"


    def sample_records(self):
        records = []
        n_rows = random.randint(20, 50)
        for _ in range(n_rows):
            record = {}

            # ID features
            id_dict = self._sample_name_username()
            for key in self.id_keys:
                record[key] = id_dict.get(key, self.faker.name())

            # Numeric features

            # for key in self.number_keys:
            #     record[key] = self._sample_number()
            for idx, key in enumerate(self.number_keys):
                record[key] = self._sample_number(offset=0.1 * idx)
            
            # Textual features
            for key in self.text_keys:
                record[key] = self._sample_text()

            records.append(record)

        # Replace one random record's name with the "special" encoding test name
        if "name" in self.id_keys and records:
            records[random.randint(0, len(records) - 1)]["name"] = self.safe_name
        
        # For each numeric key, ensure at least one is None
        for key in self.number_keys:
            if records:
                records[random.randint(0, len(records) - 1)][key] = None
                

        return records

    def write_csv(self):
        records = self.sample_records()
        df = pd.DataFrame(records)

        # Write with specified encoding, delimiter, and quotechar
        try:
            df.to_csv(
                self.path,
                encoding=self.encoding,
                sep=self.delimiter,
                quotechar=self.quotechar,
                index=False,
                header=True,
            )
        except Exception as e:
            # df.head()
            print(f"Sampled DataFrame before error:")
            print(df.head())
            print("formatting params:", self.encoding, self.delimiter, self.quotechar)
            print(f"Error writing CSV with encoding {self.encoding}: {e}")
            raise e

        # If skiprows=1, prepend a comment line to the CSV file
        if self.skiprows > 0:
            with open(self.path, 'r', encoding=self.encoding) as f:
                original_content = f.read()
            with open(self.path, 'w', encoding=self.encoding) as f:
                for _ in range(self.skiprows):
                    f.write("# This is a comment line to skip\n")
                f.write(original_content)
        

        # df.to_csv(
        #     self.path,
        #     encoding=self.encoding,
        #     sep=self.delimiter,
        #     quotechar=self.quotechar,
        #     index=False,
        #     header=True,
        # )
        # print(f"✅ CSV written: {self.path} ({self.encoding}, {self.delimiter=}, {self.quotechar=})")
        # print(f"Columns: {list(df.columns)}")

        code_to_read = (
            f"import pandas as pd\n"
            f"df = pd.read_csv(\n"
            f"    \"{{path}}\",\n"
            f"    encoding={self.encoding!r},\n"
            f"    sep={self.delimiter!r},\n"
            f"    quotechar={self.quotechar!r},\n"
            f"    skiprows={self.skiprows},\n"
            f"    header=0\n"
            f")"
        )
        meta_info = {
            "encoding": self.encoding,
            "delimiter": self.delimiter,
            "quotechar": self.quotechar,
            "skiprows": str(self.skiprows),
            "path": Path(self.path).name,
            "n_rows": len(df),
            "locale": self.locale,
            "key_names": {
                "id_keys": self.id_keys,
                "number_keys": self.number_keys,
                "text_keys": self.text_keys,
            },
            "read_code": code_to_read,
        }
        # print(f"✅ CSV written: {self.path}")
        # print(f"Columns: {list(df.columns)}")
        # print("\nTo read this file correctly, run:\n")
        # print(code_to_read)
        return df, meta_info



