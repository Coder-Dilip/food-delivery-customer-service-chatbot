# yourapp/routing.py
from django.urls import re_path

from home import consumers

websocket_urlpatterns = [
    re_path(r'ws/generate-food/$', consumers.RecipeConsumer.as_asgi()),
]
