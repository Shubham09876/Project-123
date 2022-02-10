import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import PIL.ImageOps 
import os , ssl , time
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
from sklearn.model_selection import train_test_split

# TO SET THE HTTPS CONTEXT TO FETCH THE DATA 
if ( not os.environ.get( ' PYTHONHTTPSVERIFY ' , "  " ) and getattr( ssl , " _create_unverified_context " , None ) ):
    ssl._create_default_https_context = ssl._create_unverified_context

x , y = fetch_openml("mnist_784" , version = 1 , return_X_y = True)

# TO COUNT THE NUMBER OF SAMPLES FOR EACH DIGIT
print(pd.Series(y).value_counts())

classes = ["0" , "1" , "2" , "3" , "4" , "5" , "6" , "7" , "8" , "9"]
n = len(classes)

# TO SPLIT THE DATA FOR TRAINING AND TESTING
xTrain , xTest , yTrain , yTest = train_test_split(x , y , random_state = 8 , train_size = 7500 , test_size = 2500)

# TO SCALE THE XTRAIN AND XTEST 
xTrainScale = xTrain / 255
xTestScale = xTest / 255

# TO INITIALIZE THE MODEL TO BE FOLLOWED
model = LogisticRegression( solver =  "saga" , multi_class = "multinomial".fit( xTrainScale , yTrain ) )

# TO CALCULATE THE ACCURACY OF THE MODEL
predictions = model.predict( xTestScale )
accuracy = accuracy_score( yTest , predeictions )
print( accuracy )


# TO START THE CAMERA
cam = cv2.VideoCapture( 0 )

while( True ):

    try : 
    
        # TO CAPTURE THE FRAMES AND RETURN IT
        ret , frame = cam.read()
        
        # TO CONVERT THE VIDEO INTO A GREY SCALE
        grey = cv2.cvtColor( frame , cv2.COLOR_BGR2GRAY )

        # TO DRAW A BOX IN THE CENTER OF THE VIDEO
        height , width = grey.shape
        upperLeft = ( int( width / 2 - 56 ) , int( height / 2 - 56 ) )
        bottomRight = ( int( width / 2 + 56 ) , int( height / 2 + 56 ) )

        cv2.rectangle( grey , upperLeft , bottomRight , ( 0 , 255 , 0 ) , 2 )

        # TO CONSIDER THE AREA INSIDE THE RECTANGLE CREATED
        roi = grey[ upperLeft[ 1 ] : bottomRight[ 1 ] , upperLeft[ 0 ] : bottomRight[ 0 ] ]
        
        # TO CONVERT THE CV2 IMAGES TO PIL IMAGES
        image_pil = Image.fromarray( roi )
        
        # TO CONVERT THE GREY IMAGE IN THE L FORMAT - WHERE EACH PIXEL IS ANY VALUE FROM 0 TO 255
        image_convert = image_pil.convert( "L" )
        
        # TO RESIZE THE IMAGES TO 28 BY 28 SIZE
        imageResize = image_convert.resize( ( 28 , 28 ) , Image.ANTIALIAS )

        # TO INVERT THE IMAGE 
        image_invert = PIL.ImageOps.invert( imageResze )
        
        # TO GET THE MINIMUM AND MAXIMUM PIXEL
        pixelFilter = 20
        minimumPixel = np.percentile( image_invert , pixelFilter )
        imageScale = np.clip( image_invery - minimumPixel , 0 , 255 )
        maximumPixel = np.max( image_invert )

        # TO CONVERT THE DATA INTO AN ARRAY
        imageScale = np.asarray( imageScale ) / maximumPixel
        
        # TO CREATE A SAMPLE TO MAKE A PREDICTION
        sample =  np.array( imageScale ).reshape( 1 , 784 )
        prediction = model.predict()
        print(prediction)

        # TO DISPLAY THE OUTPUT
        cv2.imshow( "Frame" , grey )

        if cv2.waitKey( 1 ) & 0xFF == ord( "q" ):
            break
    
    except Exception as e:
        pass

cam.release(  )
cv2.destroyAllWindows()