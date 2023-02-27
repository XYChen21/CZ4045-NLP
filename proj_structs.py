class Restaurant:
    def __init__(self, name: str, address: str, url: str, cuisine: str = None):
        """
        Restaurant class
        :param name: Restaurant name
        :param address: Restaurant address
        :param cuisine: Restaurant cuisine
        """

        self.name = name
        self.address = address
        self.url = url
        self.cuisine = cuisine
        self.reviews = []

    def __str__(self):
        return f"Restaurant object(name: {self.name}, addr: {self.address}, url: {self.url}, uid: {self.uid}"


class Review:
    def __init__(self, content: str, rating: int, uid: str, polarity: int = None):
        """
        Review class
        :param content: Review content
        :param rating: Review rating [0, 5]
        :param uid: Review unique identifier
        :param polarity: Review polarity, must be 0 (neg) or 1 (pos), nullable
        """
        assert polarity in [0, 1, 2, None], 'Polarity can only be 0 or 1, nullable'
        assert rating >= 0 or rating <= 5, 'Rating must be between 0 to 5, you can normalise the rating'

        self.content = content
        self.rating = rating
        self.polarity = polarity
        self.uid = uid