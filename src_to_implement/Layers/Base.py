"""
Task:
- Implement a class BaseLayer in the le Base.py in folder Layers . DONE

- This class will be inherited by every layer in our framework. For information on inheri
tance in python, please refer to here. OKAY

- Write a constructor for this class receiving no arguments. In this constructor, initialize
 a boolean member trainable with False. This member will be used to distinguish
 trainable from non-trainable layers. DONE

- Optionally, you can add other members like a default weights parameter, which might
 come in handy. DONE

"""
class BaseLayer:
    def __init__(self):
        self.trainable = False #BOOL
        self.weights = None # optional
        self.bias = None
        self.name = "BaseLayer"
        self.testing_phase =False
