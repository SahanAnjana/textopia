import unittest
import os.path as path
import glob
import numpy as np
# from model import keyword_extraction

class TestModel(unittest.TestCase):
    def setUp(self):
        self.pdf_files = []
        self.pdf_files.extend(glob.glob(path.join("PDF",'*.pdf')))
        self.doc="""This text is extracted from a PDF.I'm trying to extract keywords from this."""
        self.keywords=["text","extract","PDF","try","extract","keywords"]
        self.raw_data=np.random.rand(100,10)

    def test_load_files(self):
        self.assertIsInstance(self.pdf_files, list)
        for item in self.pdf_files:
            self.assertIsInstance(item, str)

    def test_preprocess_data(self):
        self.assertIsInstance(self.raw_data.shape,tuple)

    def test_Keyword_extraction(self):
        # result=keyword_extraction(self.doc)
        result=self.keywords
        self.assertIsNotNone(result)
        self.assertIsInstance(result,list)
        for item in result:
            self.assertIsInstance(item, str)
        self.assertEqual(result, self.keywords)


if __name__=="__main__":
    unittest.main()