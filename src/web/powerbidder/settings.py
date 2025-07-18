# powerbidder/settings.py

import os
from pathlib import Path
import environ

# 1. Initialize django-environ
env = environ.Env(
    # Set casting and default values for environment variables
    DEBUG=(bool, False) # Default DEBUG to False for security
)

# 2. Define the project's base directory
# BASE_DIR -> .../powerbidder/src/web/
BASE_DIR = Path(__file__).resolve().parent.parent

# 3. Read the .env file for local development
# This line looks for a .env file in the parent directory of this file's parent (i.e., the project root)
# It will only be used when you run `manage.py runserver` locally.
environ.Env.read_env(os.path.join(BASE_DIR.parent.parent, '.env'))

# --- Core Security Settings (Loaded from environment) ---
# This will raise an error if SECRET_KEY is not found in the environment.
SECRET_KEY = env('SECRET_KEY')

# DEBUG will be False unless you set DEBUG=True in your .env file or environment.
DEBUG = env('DEBUG')

# Load allowed hosts from a comma-separated string (e.g., "localhost,127.0.0.1,mysite.com")
ALLOWED_HOSTS = env('ALLOWED_HOSTS', default='127.0.0.1,localhost').split(',')

# This version filters out any empty strings that result from splitting an empty default.
CSRF_TRUSTED_ORIGINS = [origin for origin in env('CSRF_TRUSTED_ORIGINS', default='').split(',') if origin]


# --- Application definition ---
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # Your local apps
    'webbidder',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'powerbidder.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'powerbidder.wsgi.application'

# --- Database ---
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR.parent.parent / 'db.sqlite3', # Place db in project root
    }
}

# --- Password validation ---
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# --- Internationalization ---
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_L10N = True
USE_TZ = True

# --- Static files ---
STATIC_URL = '/static/'
# This tells Django where to look for static files in your apps
STATICFILES_DIRS = [
    BASE_DIR / 'static',
]
# We will configure STATIC_ROOT and Whitenoise in a later step.

# --- Default primary key field type ---
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# --- Static files ---
STATIC_URL = '/static/'
STATICFILES_DIRS = [
    BASE_DIR / 'static',
]

# The directory where collectstatic will gather all static files for deployment.
STATIC_ROOT = BASE_DIR.parent.parent / 'staticfiles'
# Use Whitenoise's storage backend for compression and caching.
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'