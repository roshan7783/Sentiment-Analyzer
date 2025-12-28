from youtube_comment_downloader import YoutubeCommentDownloader

def get_comments(url, limit=100):
    downloader = YoutubeCommentDownloader()
    comments = []
    for c in downloader.get_comments_from_url(url):
        comments.append(c['text'])
        if len(comments) >= limit:
            break
    return comments