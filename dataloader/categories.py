# List of actions
actions = {
    'move': {'description': 'controls the robot movement, the target object is the location where the robot will be moved to.'},
    'pick': {'description': 'controls the robot arm, the target object is the object which the robot should pick up.'},
    'finish': {'description': 'Showing that the high-level instruction task is completed. The target object does not matter thus the object prediction can be anything.'},
    'place_to': {'description': 'controls the robot arm, the target object is the location where the robot will place the grasping object to.'},
    'open': {'description': 'controls the robot arm to open an object, like a drawer or cabinet.'},
    'close': {'description': 'controls the robot arm to close an object, like a drawer or cabinet.'},
    'LongOpen': {'description': 'controls the robot arm to perform a longitudinal open movement on an object, like a sliding drawer or dresser.'},
    'LongClose': {'description': 'controls the robot arm to perform a longitudinal close movement on an object, like a sliding drawer or dresser.'},
    'RevOpen': {'description': 'controls the robot arm to perform a revolute open movement on an object, like a hinged door or cabinet.'},
    'RevClose': {'description': 'controls the robot arm to perform a revolute close movement on an object, like a hinged door or cabinet.'},
}


# actions = ['move', 'pick', 'place', 'finish']

# List of colors
color = ['red', 'yellow', 'orange', 'purple', 'blue', 'green', 'brown', 'black', 'pink', 'white']
# color = ['red', 'blue', 'green']

# list of room types
room = ['living room', 'dining room', 'bedroom', 
        'kitchen', 'bathroom',
        'house office', 'laundry room']

# list of graph types
base = ['robot', 'floor']

# list of node types
type = ['large_object', 'small_object', 'room'] + base

# List of operation objects can do. It does not involve 'move' as it is a motion of robot itself
operation = [
    'horizontal_open',
    'horizontal_close',
    'revolute_open',
    'revolute_close', 
    'longitudinal_open',
    'longitudinal_close',
    'pick',
    'place_to',
    'pour_to'] 


fruit = ['apple', 'banana', 'orange', 'grapes', 'lemon', 'strawberries', 'blueberries',
          'watermelon', 'pineapple', 'mango', 'kiwi', 'peach', 'pear', 'plum', 'cherries',
          'avocado', 'raspberries', 'blackberries', 'grapefruit', 'cantaloupe']
fruit_action = {key.lower(): ['pick'] for key in fruit}
fruit_color = {
    'apple':            'red',
    'banana':           'yellow',
    'orange':           'orange',
    'grapes':           'purple',
    'lemon':            'yellow',
    'strawberries':     'red',
    'blueberries':      'blue',
    'watermelon':       'green',
    'pineapple':        'yellow',
    'mango':            'orange',
    'kiwi':             'brown',
    'peach':            'orange',
    'pear':             'green',
    'plum':             'purple',
    'cherries':         'red',
    'avocado':          'green',
    'raspberries':      'red',
    'blackberries':     'black',
    'grapefruit':       'pink',
    'cantaloupe':       'orange'
}

object_action = {
    'bottle':           ['pick', 'pour_to'],
    'cup':              ['pick', 'pour_to'],
    'mug':              ['pick', 'pour_to'],
    'pen':              ['pick'],
    'pencil':           ['pick'],
    'remote control':   ['pick'],
    'mobile phone':     ['pick'],
    'wallet':           ['pick'],
    'glasses':          ['pick'],
    'sunglasses':       ['pick'],
    'coin':             ['pick'],
    'watch':            ['pick'],
    'hairbrush':        ['pick'],
    'lighter':          ['pick'],
    'candle':           ['pick'],
    'usb drive':        ['pick'],
    'flash drive':      ['pick'],
    'earphone':         ['pick'],
    'headphone':        ['pick'],
    'charger':          ['pick'],
    'book':             ['pick', 'place_to', 'revolute_open', 'revolute_close'],
    'notebook':         ['pick', 'place_to', 'revolute_open', 'revolute_close']
}

# The furniture that can appear in any rooms
universal_furniture_actions = {
    'console table':    ['place_to'],
    'bench':            ['place_to'], 
    'windowsill':       ['place_to'],
    'shelf':            ['place_to'],
    'chair':            ['place_to'],
    'desk':             ['place_to'],
    'rack':             ['place_to'],
    'ottoman':          ['place_to'],
    'couch':            ['place_to'],
    'bean bag':         ['place_to'],
    'box':              ['place_to']
}

# Stores openable objects that can  appear in any room
universal_openable_objects = {
    'drawer':           ['place_to', 'longitudinal_open', 'longitudinal_close'],
    'toolbox':          ['place_to', 'longitudinal_open', 'longitudinal_close'],
    'dresser':          ['place_to', 'longitudinal_open', 'longitudinal_close'],
    'side table':       ['place_to', 'longitudinal_open', 'longitudinal_close'],
    'storage cabinet':  ['place_to', 'revolute_open', 'revolute_close'],
    'medicine cabinet': ['place_to', 'revolute_open', 'revolute_close'],
    'briefcase':        ['place_to', 'revolute_open', 'revolute_close'],
}

furniture_action = {
    'dressing table':   ['place_to'],             
    'tv stand':         ['place_to'],
    'sofa':             ['place_to'],
    'bed':              ['place_to'],
    'bookshelf':        ['place_to'],    
    'bar stool':        ['place_to'],
    'kitchen island':   ['place_to'],
    'dining table':     ['place_to'],
    'display shelves':  ['place_to'],
    'conference table': ['place_to'],
    'file organizer':   ['place_to'],
    'vanity sink':      ['place_to'],
    'washstand':        ['place_to'],
    'freestanding rack':['place_to'],
    'basket':           ['place_to'],
    'window':           [''],
    'coffee table':     ['place_to', 'longitudinal_open', 'longitudinal_close'],
    'bookcase':         ['place_to', 'revolute_open', 'revolute_close'],
}

# Define necessary objects for each room type
room_required_objects = {
    'kitchen': ['bar stool', 'kitchen island'],
    'living room': ['sofa', 'tv stand', 'coffee table'],
    'bedroom': ['bed', 'dressing table', 'window'],
    'dining room': ['dining table', 'display shelves'],
    'bathroom': ['vanity sink', 'washstand'],
    'house office': ['bookcase', 'conference table', 'file organizer'],
    'laundry room': ['freestanding rack', 'basket']
}

def extract_openable_objects(*dictionaries):
    openable_objects = {}
    for dictionary in dictionaries:
        for key, actions in dictionary.items():
            if 'longitudinal_open' in actions or 'revolute_open' in actions:
                openable_objects[key] = actions
    return openable_objects

# Stores all the openable objects, only used to check whether an object is openable
openable_objects = extract_openable_objects(
    universal_furniture_actions,
    furniture_action,
    universal_openable_objects
)

large_object_dictionary = {**universal_furniture_actions, **furniture_action, **universal_openable_objects}

# Add attributes to each object in room_required_objects
for room_name, objects in room_required_objects.items():
    room_required_objects[room_name] = {object: (large_object_dictionary[object] if object in furniture_action else openable_objects[object]) for object in objects}

# Get small objects and properties
small_object = []
small_object.extend(object_action.keys()) # assign small object names
small_object.extend(fruit) # assign fruit names
small_object = [x.lower() for x in small_object] # convert to lower case

# Get large objects and properties
universal_large_object = []
universal_large_object.extend(universal_furniture_actions.keys()) # assign furniture names to large objects
universal_large_object.extend(universal_openable_objects.keys())
universal_large_object = [x.lower() for x in universal_large_object] # convert to lower case

room_required_large_object = []
room_required_large_object.extend(furniture_action.keys()) # assign other furniture names to large objects
room_required_large_object = [x.lower() for x in room_required_large_object] # convert to lower case

universal_openable_object = []
universal_openable_object.extend(universal_openable_objects.keys()) # assign other furniture names to large objects
universal_openable_object = [x.lower() for x in universal_openable_objects] # convert to lower case

large_object = []
large_object.extend(universal_furniture_actions)
large_object.extend(room_required_large_object) 
large_object.extend(universal_openable_objects) 

# get color of all objects
object_color = fruit_color.copy() # assign fruit color
object_color.update({key.lower(): '' for key in object_action.keys()}) # assign empty color to small objects
object_color.update({key.lower(): '' for key in universal_furniture_actions.keys()}) # assign empty color to large objects

# get operation of all objects
object_operation = {}
object_operation.update(fruit_action)
object_operation.update(object_action)
object_operation.update(universal_furniture_actions)
object_operation.update(furniture_action)
object_operation.update(universal_openable_objects)


# add items to dictionary
categories = {}
# Type of operations
categories['operation']=operation
# Type of colors
categories['color']=color

# The subsets of label
categories['room']=room
categories['small_object']=small_object
categories['large_object']=large_object
# Universal is a subset of large_object
categories['universal_large_object']=universal_large_object


# contains properties of all objects. The key is the label name
categories['object_color']=object_color # color of objects, if it is empty, any color can be used 
categories['object_operation']=object_operation # available operation for objects

# All label names
categories['label'] = categories['room']+categories['small_object']+categories['large_object']

# All type names
categories['type']=type

if __name__ =="__main__":
    print(categories['label'])
    print(categories['type'])
    print(categories['color'])