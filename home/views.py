from django.shortcuts import redirect, render
import google.generativeai as genai
from decouple import config
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
# Configure the SDK with your API key
genai.configure(api_key=config('GOOGLE_API_KEY'))
def recipeGeneration(req):
    return render(req,'home/recipe_generation.html')


@csrf_exempt
def classify_query(request):
    if request.method == 'POST':
        query = request.POST.get('query', '')

        if query:
            model_name = 'gemini-1.0-pro'  # For text-based tasks
            prompt = f"You have to do zero shot classification for this sentence: {query} from these classes: 'Payment', 'Cancel Order','Food Queries'. Just return me only that class name nothing else. for example the output can be 'payment'"
            
            # Initialize the Generative Model
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            
            # Extract and return the response text
            result = response.text.strip()
            return JsonResponse({'class_name': result})
        
        return JsonResponse({'error': 'No query provided'}, status=400)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)



        




