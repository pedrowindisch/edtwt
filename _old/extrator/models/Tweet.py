from dataclasses import dataclass


@dataclass(slots=True)
class Tweet:
    id: str
    url: str
    text: str
    source: str
    created_at: str
    lang: str
    retweet_count: int
    reply_count: int
    like_count: int
    quote_count: int
    view_count: int
    bookmark_count: int
    conversation_id: str
    in_reply_to_id: str
    in_reply_to_user_id: str
    in_reply_to_username: str
    is_reply: bool
    is_limited_reply: bool
    author_id: str
    author_name: str
    author_username: str
    author_url: str
    author_followers: int
    author_following: int
    hashtags: list[str]
    urls: list[str]
    mentions: list[str]

    @classmethod
    def from_payload(cls, payload: dict) -> "Tweet":
        author = payload.get("author") or {}
        entities = payload.get("entities") or {}

        hashtags = [normalize_text(item.get("text")) for item in entities.get("hashtags", [])]
        urls = [normalize_text(item.get("expanded_url") or item.get("url")) for item in entities.get("urls", [])]
        mentions = [normalize_text(item.get("screen_name")) for item in entities.get("user_mentions", [])]

        return cls(
            id=normalize_text(payload.get("id")),
            url=normalize_text(payload.get("url")),
            text=normalize_text(payload.get("text")),
            source=normalize_text(payload.get("source")),
            created_at=normalize_text(payload.get("createdAt")),
            lang=normalize_text(payload.get("lang")),
            retweet_count=int(payload.get("retweetCount", 0) or 0),
            reply_count=int(payload.get("replyCount", 0) or 0),
            like_count=int(payload.get("likeCount", 0) or 0),
            quote_count=int(payload.get("quoteCount", 0) or 0),
            view_count=int(payload.get("viewCount", 0) or 0),
            bookmark_count=int(payload.get("bookmarkCount", 0) or 0),
            conversation_id=normalize_text(payload.get("conversationId")),
            in_reply_to_id=normalize_text(payload.get("inReplyToId")),
            in_reply_to_user_id=normalize_text(payload.get("inReplyToUserId")),
            in_reply_to_username=normalize_text(payload.get("inReplyToUsername")),
            is_reply=bool(payload.get("isReply", False)),
            is_limited_reply=bool(payload.get("isLimitedReply", False)),
            author_id=normalize_text(author.get("id")),
            author_name=normalize_text(author.get("name")),
            author_username=normalize_text(author.get("userName")),
            author_url=normalize_text(author.get("url")),
            author_followers=int(author.get("followers", 0) or 0),
            author_following=int(author.get("following", 0) or 0),
            hashtags=[item for item in hashtags if item],
            urls=[item for item in urls if item],
            mentions=[item for item in mentions if item],
        )

    @staticmethod
    def csv_headers() -> list[str]:
        return [
            "search_date",
            "id",
            "url",
            "text",
            "source",
            "created_at",
            "lang",
            "retweet_count",
            "reply_count",
            "like_count",
            "quote_count",
            "view_count",
            "bookmark_count",
            "conversation_id",
            "in_reply_to_id",
            "in_reply_to_user_id",
            "in_reply_to_username",
            "is_reply",
            "is_limited_reply",
            "author_id",
            "author_name",
            "author_username",
            "author_url",
            "author_followers",
            "author_following",
            "hashtags",
            "urls",
            "mentions",
        ]

    def to_record(self, search_date: str) -> dict[str, str | int | bool]:
        return {
            "search_date": search_date,
            "id": self.id,
            "url": self.url,
            "text": self.text,
            "source": self.source,
            "created_at": self.created_at,
            "lang": self.lang,
            "retweet_count": self.retweet_count,
            "reply_count": self.reply_count,
            "like_count": self.like_count,
            "quote_count": self.quote_count,
            "view_count": self.view_count,
            "bookmark_count": self.bookmark_count,
            "conversation_id": self.conversation_id,
            "in_reply_to_id": self.in_reply_to_id,
            "in_reply_to_user_id": self.in_reply_to_user_id,
            "in_reply_to_username": self.in_reply_to_username,
            "is_reply": self.is_reply,
            "is_limited_reply": self.is_limited_reply,
            "author_id": self.author_id,
            "author_name": self.author_name,
            "author_username": self.author_username,
            "author_url": self.author_url,
            "author_followers": self.author_followers,
            "author_following": self.author_following,
            "hashtags": "|".join(self.hashtags),
            "urls": "|".join(self.urls),
            "mentions": "|".join(self.mentions),
        }

    def to_csv_row(self, search_date: str) -> dict[str, str | int | bool]:
        return self.to_record(search_date)


def normalize_text(value: object) -> str:
    if value is None:
        return ""

    return str(value)
