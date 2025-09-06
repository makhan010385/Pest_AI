from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import numpy as np
import os
import random
from werkzeug.utils import secure_filename
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'xlsx', 'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Sample disease data
diseases = {
    'leaf_blight': {
        'name': 'Leaf Blight',
        'description': 'A fungal disease causing brown spots and lesions on leaves.',
        'treatment': 'Apply copper-based fungicides. Remove affected leaves and improve air circulation.',
        'prevention': 'Avoid overhead watering, ensure proper spacing between plants, and rotate crops annually.'
    },
    'powdery_mildew': {
        'name': 'Powdery Mildew',
        'description': 'White powdery coating on leaves and stems caused by fungal infection.',
        'treatment': 'Use sulfur-based fungicides or neem oil. Prune affected areas.',
        'prevention': 'Ensure good air circulation, avoid overcrowding, and water at soil level.'
    },
    'bacterial_spot': {
        'name': 'Bacterial Spot',
        'description': 'Dark spots with yellow halos on leaves caused by bacterial infection.',
        'treatment': 'Apply copper-based bactericides. Remove infected plant material.',
        'prevention': 'Use disease-free seeds, avoid overhead irrigation, and practice crop rotation.'
    },
    'rust': {
        'name': 'Rust',
        'description': 'Orange or reddish pustules on leaves caused by fungal infection.',
        'treatment': 'Apply systemic fungicide and remove infected plant parts.',
        'prevention': 'Avoid overhead watering, plant resistant varieties, and ensure good air circulation.'
    },
    'healthy': {
        'name': 'Healthy Plant',
        'description': 'No disease detected. Plant appears healthy.',
        'treatment': 'Continue current care practices. Monitor regularly for any changes.',
        'prevention': 'Maintain good cultural practices, proper nutrition, and regular monitoring.'
    }
}

# Sample pest data
pests = {
    'aphids': {
        'name': 'Aphids',
        'description': 'Small, soft-bodied insects that feed on plant sap.',
        'treatment': 'Use insecticidal soap, neem oil, or introduce beneficial insects like ladybugs.',
        'prevention': 'Encourage beneficial insects, avoid over-fertilizing with nitrogen, and use reflective mulches.'
    },
    'whiteflies': {
        'name': 'Whiteflies',
        'description': 'Tiny white flying insects that cluster on leaf undersides.',
        'treatment': 'Use yellow sticky traps, insecticidal soap, or systemic insecticides.',
        'prevention': 'Remove weeds, use reflective mulches, and inspect plants regularly.'
    },
    'spider_mites': {
        'name': 'Spider Mites',
        'description': 'Microscopic pests causing stippled, yellowing leaves.',
        'treatment': 'Increase humidity, use miticides, or spray with water to dislodge mites.',
        'prevention': 'Maintain adequate moisture, avoid dusty conditions, and use predatory mites.'
    },
    'thrips': {
        'name': 'Thrips',
        'description': 'Tiny, slender insects with fringed wings causing silvery streaks.',
        'treatment': 'Use blue sticky traps, predatory mites, or appropriate insecticides.',
        'prevention': 'Remove weeds, use beneficial nematodes, and maintain proper humidity.'
    },
    'caterpillars': {
        'name': 'Caterpillars',
        'description': 'Larvae of moths/butterflies that chew leaves and bore into stems.',
        'treatment': 'Hand picking, Bt (Bacillus thuringiensis), or targeted insecticides.',
        'prevention': 'Row covers, beneficial wasps, companion planting, and regular monitoring.'
    },
    'no_pest': {
        'name': 'No Pest Detected',
        'description': 'No harmful pests identified in the image.',
        'treatment': 'Continue monitoring. Maintain current pest management practices.',
        'prevention': 'Regular scouting, encourage beneficial insects, and practice IPM strategies.'
    }
}

# Weed data
weeds = {
    'dandelion': {
        'name': 'Dandelion',
        'description': 'Yellow flowers with deeply toothed leaves and deep taproot.',
        'treatment': 'Hand pulling when soil is moist, or use selective herbicide.',
        'prevention': 'Maintain thick turf, regular mowing, and corn gluten meal application.'
    },
    'crabgrass': {
        'name': 'Crabgrass',
        'description': 'Low-growing grass that spreads by seeds and crowds out desirable grass.',
        'treatment': 'Pre-emergent herbicide in early spring or hand removal.',
        'prevention': 'Dense turf, proper fertilization, and overseeding in fall.'
    },
    'clover': {
        'name': 'Clover',
        'description': 'Three-leaflet leaves with white or pink flowers.',
        'treatment': 'Selective herbicide or hand removal if unwanted.',
        'prevention': 'Improve soil fertility and maintain dense turf.'
    },
    'no_weed': {
        'name': 'No Weed Detected',
        'description': 'No problematic weeds identified.',
        'treatment': 'Continue current maintenance practices.',
        'prevention': 'Maintain healthy soil and plant density.'
    }
}

def get_chatbot_response(message):
    """Enhanced chatbot for agricultural queries"""
    message = message.lower()
    
    # Disease-related queries
    if any(word in message for word in ['disease', 'sick', 'spots', 'yellowing', 'wilting', 'fungus']):
        return {
            'text': "I can help you identify plant diseases! Common symptoms include:\n\n• **Leaf spots**: Often caused by fungal infections\n• **Yellowing**: Could indicate nutrient deficiency or overwatering\n• **Wilting**: May be due to root problems or water stress\n\nFor accurate diagnosis, please upload a clear photo using our Disease Classification tool.",
            'suggestions': ['Upload disease photo', 'Common plant diseases', 'Prevention tips']
        }
    
    # Pest-related queries
    elif any(word in message for word in ['pest', 'bug', 'insect', 'aphid', 'caterpillar', 'mite']):
        return {
            'text': "Pest problems can seriously damage crops! Here are common signs:\n\n• **Holes in leaves**: Often caused by caterpillars or beetles\n• **Sticky honeydew**: Usually from aphids or whiteflies\n• **Stippled leaves**: Typically spider mite damage\n\nUse our Pest Detection tool for specific identification and treatment.",
            'suggestions': ['Upload pest photo', 'Organic pest control', 'Beneficial insects']
        }
    
    # Weed-related queries
    elif any(word in message for word in ['weed', 'grass', 'unwanted', 'invasive']):
        return {
            'text': "Weed management is crucial for healthy crops! Effective strategies:\n\n• **Prevention**: Maintain dense, healthy plant cover\n• **Early detection**: Regular monitoring and quick action\n• **Integrated approach**: Combine cultural, mechanical, and biological methods\n\nOur Weed Management tool can help identify and control weeds.",
            'suggestions': ['Upload weed photo', 'Organic weed control', 'Prevention strategies']
        }
    
    # Nutrition queries
    elif any(word in message for word in ['fertilizer', 'nutrition', 'nutrient', 'npk', 'compost']):
        return {
            'text': "Proper nutrition is essential! Key nutrients:\n\n• **Nitrogen (N)**: Leafy growth and green color\n• **Phosphorus (P)**: Root development and flowering\n• **Potassium (K)**: Disease resistance and fruit quality\n\nSoil testing is recommended before fertilizing.",
            'suggestions': ['Soil testing', 'Organic fertilizers', 'Nutrient deficiency signs']
        }
    
    # Watering queries
    elif any(word in message for word in ['water', 'irrigation', 'drought', 'overwater']):
        return {
            'text': "Proper watering tips:\n\n• **Deep, infrequent watering** is better than frequent shallow watering\n• **Morning watering** reduces disease risk\n• **Mulching** helps retain soil moisture\n• **Check soil moisture** before watering",
            'suggestions': ['Watering schedule', 'Mulching tips', 'Drought management']
        }
    
    # Default response
    else:
        responses = [
            {
                'text': "Hello! I'm your agricultural assistant. I can help with:\n\n• **Plant disease diagnosis** and treatment\n• **Pest identification** and management\n• **Weed control** strategies\n• **General farming advice**\n\nWhat would you like to know about?",
                'suggestions': ['Disease help', 'Pest control', 'Weed management', 'Farming tips']
            }
        ]
        return random.choice(responses)

# Routes from original app.py
@app.route('/')
def index():
    """Main dashboard with overview of all agricultural modules"""
    return render_template('index.html')

@app.route('/disease-classification')
def disease_classification():
    """Disease classification module for crop diseases"""
    return render_template('disease_classification.html')

@app.route('/pest-detection')
def pest_detection():
    """Insect and pest detection module"""
    return render_template('pest_simple.html')

@app.route('/weed-management')
def weed_management():
    """Weed identification and management module"""
    return render_template('weed_management.html')

@app.route('/chatbot')
def chatbot():
    """Agricultural chatbot for farmers"""
    return render_template('chatbot.html')

@app.route('/pest-analysis')
def pest_analysis():
    """Advanced pest analysis using existing data"""
    return render_template('pest_analysis.html')

# Simple routes for minimal templates
@app.route('/disease')
def disease_simple():
    return render_template('disease_simple.html')

@app.route('/pest')
def pest_simple():
    return render_template('pest_simple.html')

@app.route('/chat-simple')
def chatbot_simple():
    return render_template('chatbot_simple.html')

@app.route('/upload-image', methods=['POST'])
def upload_image():
    """Handle image uploads for disease/pest/weed detection"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image based on the module
        module = request.form.get('module', 'disease')
        result = process_image(filepath, module)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify(result)
    
    return jsonify({'error': 'Invalid file type'}), 400

def process_image(filepath, module):
    """Process uploaded image for classification/detection"""
    try:
        # Verify it's a valid image
        with Image.open(filepath) as img:
            img.verify()
        
        if module == 'disease':
            disease_key = random.choice(list(diseases.keys()))
            result = diseases[disease_key].copy()
            result['confidence'] = random.uniform(75, 95)
            result['module'] = 'Disease Classification'
            return result
        
        elif module == 'pest':
            pest_key = random.choice(list(pests.keys()))
            result = pests[pest_key].copy()
            result['confidence'] = random.uniform(70, 90)
            result['module'] = 'Pest Detection'
            return result
        
        elif module == 'weed':
            weed_key = random.choice(list(weeds.keys()))
            result = weeds[weed_key].copy()
            result['confidence'] = random.uniform(65, 88)
            result['module'] = 'Weed Management'
            return result
            
    except Exception as e:
        return {'error': f'Error processing image: {str(e)}'}

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Unified analysis endpoint for simple templates"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    analysis_type = request.form.get('type', 'disease')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, JPEG, or GIF'}), 400
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Verify it's a valid image
        with Image.open(filepath) as img:
            img.verify()
        
        # Simulate analysis based on type
        if analysis_type == 'disease':
            disease_key = random.choice(list(diseases.keys()))
            result = diseases[disease_key].copy()
            result['confidence'] = random.uniform(75, 95)
        elif analysis_type == 'pest':
            pest_key = random.choice(list(pests.keys()))
            result = pests[pest_key].copy()
            result['confidence'] = random.uniform(70, 90)
        else:  # weed
            weed_key = random.choice(list(weeds.keys()))
            result = weeds[weed_key].copy()
            result['confidence'] = random.uniform(65, 88)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify(result)
        
    except Exception as e:
        # Clean up file if it exists
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chatbot conversations"""
    data = request.get_json()
    message = data.get('message', '')
    response = get_chatbot_response(message)
    return jsonify({'response': response})

@app.route('/api/pest-data')
def get_pest_data():
    """API endpoint to get pest analysis data"""
    try:
        # Load the existing pest data
        df = pd.read_excel('popultion dynamics Book1.xlsx')
        
        # Basic statistics
        stats = {
            'total_records': len(df),
            'date_range': {
                'start': str(df['Year'].min()) if 'Year' in df.columns else 'N/A',
                'end': str(df['Year'].max()) if 'Year' in df.columns else 'N/A'
            },
            'avg_catch': float(df['Trap Flies Catch'].mean()) if 'Trap Flies Catch' in df.columns else 0,
            'max_catch': float(df['Trap Flies Catch'].max()) if 'Trap Flies Catch' in df.columns else 0
        }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
