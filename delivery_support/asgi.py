# asgi.py
import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
import home.routing

# Set the DJANGO_SETTINGS_MODULE environment variable
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'delivery_support.settings')



# Get the ASGI application for HTTP requests
django_asgi_application = get_asgi_application()

# Define the ASGI application for WebSocket connections
application = ProtocolTypeRouter({
    "http": django_asgi_application,  # HTTP handling
    "websocket": URLRouter(  # WebSocket handling
        home.routing.websocket_urlpatterns
    ),
})
