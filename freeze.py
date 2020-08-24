from flask_frozen import Freezer
from My_web import app

freezer = Freezer(app)

if __name__ == '__main__':
    freezer.freeze()