from django.urls import path,include
from . import views

from .views import classify_query


urlpatterns = [

    path('',views.recipeGeneration),
        path('api/classify-query/', classify_query, name='classify_query'),

   
]