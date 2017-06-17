from Utilities import *
from TrainingArchi import *
from TestingArchi import *

def RunApp(FirtsRun):
  
    ImageName = input(" Enter the image name : ")
    print("")
    print(" loading ...")
    print("")
    TestingImage(ImageName,FirtsRun)
    print(" Colored Image is saved in the same DIR.")
    print("")
    Desire = input(" Try Again ? Y/N ? ")
    print("")
    if (Desire == 'Y' or Desire == 'y'):
        print("")
        RunApp(True)


print("")
print("")

print("           ---- DEEP LEARNING COLORIZATION FOR VISUAL MEDIA ----           ")

print("")
print("")

RunApp(False)
