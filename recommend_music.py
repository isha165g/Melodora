import spotipy
from spotipy.oauth2 import SpotifyOAuth
import random
import webbrowser

# ================================
# SPOTIFY AUTHENTICATION
# ================================
CLIENT_ID = "7a48c156e86a4b85a9f984ab24fc5b1f"
CLIENT_SECRET = "212f6287c22d453c96f02460b493fad2"
REDIRECT_URI = "http://127.0.0.1:8080/callback"
SCOPE = "playlist-modify-public"

# Authenticate with Spotify
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE,
    show_dialog=True,
    cache_path=None
))

# ================================
# EMOTION → PROGRESSION MAP
# ================================
# Each emotion transitions gradually to positive mood
emotion_progression = {
    "Sad": ["sad acoustic", "hopeful pop", "happy upbeat"],
    "Angry": ["angry rock", "lofi chill", "peaceful acoustic"],
    "Fear": ["calming piano", "motivational soft", "confident pop"],
    "Disgust": ["cleanse ambient", "peaceful chill", "positive energy"],
    "Neutral": ["soft pop", "feel good", "happy hits"],
    "Happy": ["energetic pop", "dance party", "cool down acoustic"],
    "Surprise": ["exciting edm", "curious indie", "steady vibe"]
}

# ================================
# DYNAMIC PLAYLIST GENERATOR
# ================================
def recommend_playlist(emotion):
    """Dynamically create a playlist that transitions to a better emotional state."""
    print(f"\n🧠 Detected Emotion: {emotion}")

    if emotion not in emotion_progression:
        print("⚠️ No progression mapping found for this emotion.")
        return

    # Use emotion stages (start → mid → positive)
    stages = emotion_progression[emotion]
    all_tracks = []

    for stage in stages:
        print(f"🎧 Searching for songs with vibe: {stage}")
        results = sp.search(q=stage, type='track', limit=10)

        for track in results['tracks']['items']:
            all_tracks.append(track['uri'])

    # Avoid duplicates
    all_tracks = list(dict.fromkeys(all_tracks))

    if not all_tracks:
        print("❌ No tracks found for this emotion.")
        return

    # Create playlist in user account
    user_id = sp.me()['id']
    playlist_name = f"Melodora - {emotion} Mood Journey 🌈"
    playlist = sp.user_playlist_create(user=user_id, name=playlist_name, public=True)
    playlist_id = playlist['id']

    # Add tracks
    sp.playlist_add_items(playlist_id, all_tracks[:30])

    # Get playlist URL
    playlist_url = playlist['external_urls']['spotify']
    print(f"✅ Playlist created successfully: {playlist_url}")

    # Open in browser
    webbrowser.open(playlist_url)
