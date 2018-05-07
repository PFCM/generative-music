"""
Tests for the data module
"""
import unittest

import drums.data as data


class NoteTests(unittest.TestCase):
    """tests for the note packing and unpacking"""

    pack_cases = {
        0: data.Note(0, 0, 0, 0),
        5: data.Note(5, 0, 0, 0),
        0b00110001000000101: data.Note(5, 0, 16, 24),
        0xffff: data.Note(7, 3, 31, 63)
    }

    def test_packs_expected(self):
        """check a few obvious ones pack"""
        for packed, unpacked in self.pack_cases.items():
            self.assertEqual(packed, unpacked.to_int())

    def test_unpacks_expected(self):
        """check a few obvious ones unpack"""
        for packed, unpacked in self.pack_cases.items():
            self.assertEqual(unpacked, data.Note.from_int(packed))

    def test_pack_unpack(self):
        """check if we pack something it unpacks to the same thing"""
        for packed, unpacked in self.pack_cases.items():
            self.assertEqual(packed, data.Note.from_int(packed).to_int())
            self.assertEqual(unpacked, data.Note.from_int(unpacked.to_int()))
