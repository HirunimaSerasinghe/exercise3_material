# Important changes to remember

#### line 1604 NeuralNetworkTests.py changed
        #self.assertNotEqual(out, out2)
        self.assertFalse(np.array_equal(out, out2))