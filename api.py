from atproto import Client, models
from typing import Dict, List, Optional
import json
import os
from datetime import datetime

class PublicBlueskyAPI:
    """Class for handling public Bluesky API endpoints that don't require authentication"""
    
    def __init__(self):
        self.client = Client("https://public.api.bsky.app")

    def resolve_handle(self, handle: str) -> str:
        """Convert a handle (e.g., natolambert.bsky.social) to a DID"""
        return self.client.com.atproto.identity.resolve_handle({'handle': handle}).did

    def get_profile(self, handle: str) -> Dict:
        """Get public profile information for a user"""
        return self.client.app.bsky.actor.get_profile({'actor': handle}).model_dump()

    def get_posts(self, handle: str, limit: int = 100) -> List[Dict]:
        """Get public posts from a specific user"""
        posts = []

        cursor = None
        while True:
            params = models.AppBskyFeedGetAuthorFeed.Params(
                actor=handle,
                filter="posts_no_replies",
                limit=min(100, limit - len(posts)),
                cursor=cursor
            )
            response = self.client.app.bsky.feed.get_author_feed(params)
            posts.extend([post.model_dump() for post in response.feed])

            if not response.cursor or len(posts) >= limit:
                break
            cursor = response.cursor

        return posts[:limit]

    def get_follows(self, handle: str, limit: int = 100) -> List[Dict]:
        """Get public list of users that this user follows"""
        follows = []
        cursor = None
        while True:
            params = models.AppBskyGraphGetFollows.Params(
                actor=handle,
                limit=min(100, limit - len(follows)),
                cursor=cursor
            )
            response = self.client.app.bsky.graph.get_follows(params)
            follows.extend([follow.model_dump() for follow in response.follows])

            if not response.cursor or len(follows) >= limit:
                break
            cursor = response.cursor

        return follows[:limit]

    def get_followers(self, handle: str, limit: int = 100) -> List[Dict]:
        """Get public list of users that follow this user"""
        followers = []
        cursor = None

        while True:
            params = models.AppBskyGraphGetFollowers.Params(
                actor=handle,
                limit=min(100, limit - len(followers)),
                cursor=cursor
            )
            response = self.client.app.bsky.graph.get_followers(params)
            followers.extend([follower.model_dump() for follower in response.followers])

            if not response.cursor or len(followers) >= limit:
                break
            cursor = response.cursor

        return followers[:limit]

class BlueskyAPI(PublicBlueskyAPI):
    """Class for handling both public and authenticated Bluesky API endpoints"""
    
    def __init__(self):
        super().__init__()

    def create_session(self, identifier: str, password: str) -> None:
        """Create a session using username/password"""
        self.client.login(identifier, password)

    def get_user_likes(self, handle: str, limit: int = 100) -> List[Dict]:
        """Get posts liked by this user (requires authentication)"""
        if not self.client.session:
            raise Exception("Authentication required for getting user likes")
            
        likes = []
        cursor = None

        while True:
            try:
                response = self.client.app.bsky.feed.get_actor_likes(
                    actor=handle,
                    limit=min(100, limit - len(likes)),
                    cursor=cursor
                )
                likes.extend([like.model_dump() for like in response.feed])

                if not response.cursor or len(likes) >= limit:
                    break
                cursor = response.cursor
            except Exception as e:
                if "Profile not found" in str(e):
                    return []
                raise e

        return likes[:limit]

class FirehoseSubscriber:
    def __init__(self):
        self.client = Client()

    async def process_messages(self, save_dir: Optional[str] = None):
        """Process messages from the firehose"""
        async for commit in self.client.sync.subscribe_repos():
            try:
                for op in commit.ops:
                    path = op.path
                    
                    # Process different types of interactions
                    if 'app.bsky.feed.post' in path:
                        print(f"New post from {commit.repo}")
                        if save_dir:
                            self.save_interaction('posts', op.model_dump(), save_dir)
                            
                    elif 'app.bsky.feed.like' in path:
                        print(f"New like from {commit.repo}")
                        if save_dir:
                            self.save_interaction('likes', op.model_dump(), save_dir)
                            
                    elif 'app.bsky.graph.follow' in path:
                        print(f"New follow from {commit.repo}")
                        if save_dir:
                            self.save_interaction('follows', op.model_dump(), save_dir)

            except Exception as e:
                print(f"Error processing message: {e}")
                continue

    @staticmethod
    def save_interaction(interaction_type: str, data: Dict, save_dir: str):
        """Save interaction data to a file"""
        filename = f"{save_dir}/{interaction_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(save_dir, exist_ok=True)
        
        with open(filename, 'a') as f:
            json.dump(data, f)
            f.write('\n')
