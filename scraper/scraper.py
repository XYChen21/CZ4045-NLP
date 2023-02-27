import csv
import pickle
import urllib
from datetime import datetime
from telnetlib import EC
from time import sleep

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from db.database import Database
from proj_structs import Restaurant, Review


class BaseScraper:
    name = "Undefined"

    def __init__(self):
        self.restaurant_list = []

    def scrape(self) -> [Restaurant]:
        raise NotImplementedError

    def save_restaurants_to_db(self):
        assert len(self.restaurant_list) != 0, "Please run scrape() first before saving to db."
        db = Database()
        for restaurant in self.restaurant_list:
            db.insert_to_restaurant(restaurant)

    def save_reviews_to_csv(self):
        assert len(self.restaurant_list) != 0, "Please run scrape() first before saving to csv."
        with open('reviews_raw.csv', 'w', newline='', encoding="utf-8-sig") as file:
            writer = csv.writer(file)

            writer.writerow(["uid", "restaurant name", "rating", "content"])

            for restaurant in self.restaurant_list:
                for review in restaurant.reviews:
                    writer.writerow([review.uid, restaurant.name, review.rating, review.content])


class GoogleReviewScraper(BaseScraper):
    name = "Google Review Scraper"

    def __init__(self, headless=True):
        super().__init__()

        # init chrome driver
        options = Options()

        if headless:
            options.add_argument('--headless')

        options.add_argument('--disable-gpu')
        options.add_argument("--lang=en")
        service = Service(executable_path=ChromeDriverManager().install())
        self.driver = webdriver.Chrome(options=options, service=service)
        self.wait = WebDriverWait(self.driver, 10)

    def __del__(self):
        print("Scraper shutting down...")
        self.driver.close()  # Close the browser

    def scrape(self) -> [Restaurant]:
        limit_per_restaurant = 20
        cuisine_list = [
            "Chinese",
            "Malay", "Indian", "Japanese", "Thai", "Western"]

        town_areas = [
            # "Nanyang Technological University",
            # "Toa Payoh",
            # "Ang Mo Kio",
            # "Tanjong Pagar",
            # "Yishun",
            # "Sengkang",
            "Novena"
        ]

        for area in town_areas:
            for cuisine in cuisine_list:
                self.__scrape_restaurant(cuisine, area, limit_per_restaurant)

        self.save_restaurants_to_db()

        self.__scrape_reviews(self.restaurant_list, 50)

        with open('saved_1.pickle', 'wb+') as handle:
            pickle.dump(self.restaurant_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # with open('filename.pickle', 'rb') as handle:
        #     self.restaurant_list = pickle.load(handle)

        self.save_reviews_to_csv()
        # for i in restaurants:
        #     print(i)

    def __scrape_restaurant(self, cuisine, location, limit=20) -> [Restaurant]:
        location_parsed = urllib.parse.quote_plus(location)
        q = f"{cuisine}+restaurants+near+{location_parsed}"
        self.driver.get(f"https://www.google.com/maps/search/{q}/")

        sleep(4)  # wait
        try:
            scrollable_div = self.driver.find_element(By.CSS_SELECTOR,
                                                  f'div[aria-label="Results for {cuisine} restaurants near {location}"]')
        except Exception as ex:
            # No results returned for given query
            print("No results returned for given query")
            return []

        # Scroll through the restaurant list
        restaurant_list = []

        for i in range(10):
            self.driver.execute_script(
                'arguments[0].scrollTop = arguments[0].scrollHeight',
                scrollable_div
            )
            restaurant_list = scrollable_div.find_elements(By.CSS_SELECTOR, "div[role='article'] > a")
            # restaurant_list = scrollable_div.find_elements(By.CSS_SELECTOR, "input[data-js-log-root='']")
            if len(restaurant_list) >= limit:
                # Have enough restaurant loaded
                break

            if i % 3 == 0:
                sleep(2)
                scrollable_div.send_keys(Keys.PAGE_UP)
                # self.driver.execute_script("scroll(0, -250);", scrollable_div)

            sleep(2)

        res = []
        # go through each restaurant
        for restaurant_el in restaurant_list:
            print("click restaurant")

            try:
                restaurant_el.click()
            except Exception:
                print("Can't click restaurant, skipping...")
                continue

            # Get restaurant name, address and url
            sleep(2)
            try:
                share_btn = self.driver.find_element(By.CSS_SELECTOR,
                                                     'button[jsaction="pane.placeActions.share;keydown:pane.placeActions.share"]')
                print("click share btn")
                share_btn.click()

                url_section = self.wait.until(
                    lambda x: x.find_element(By.CSS_SELECTOR, "input[jsaction='pane.copyLink.clickInput']"))
                restaurant_url = url_section.get_attribute("value")
                restaurant_name = self.driver.find_element(By.CSS_SELECTOR, "div[jsan='7.TDF87d']").text
                restaurant_addr = self.driver.find_element(By.CSS_SELECTOR, "div[jsan='7.vKmG2c']").text
                print(restaurant_name, restaurant_addr, restaurant_url)

                r = Restaurant(
                    name=restaurant_name,
                    address=restaurant_addr,
                    url=restaurant_url,
                    cuisine=cuisine
                )
                res.append(r)
                self.restaurant_list.append(r)

            except Exception as ex:
                # skip
                self.driver.execute_script('el = document.elementFromPoint(47, 100); el.click();')
                print("ERROR! skipping")
            finally:
                close_btn = self.driver.find_element(By.CSS_SELECTOR, "button[aria-label='Close']")
                close_btn.click()
                sleep(3)

        print('done')

        # Get link
        return res

    def __scrape_reviews(self, restaurants: [Restaurant], limit=20):

        for restaurant in restaurants:
            self.driver.get(restaurant.url)
            try:
                review_btn = self.wait.until(lambda x: x.find_element(By.CSS_SELECTOR, "button[jsaction='pane.rating.moreReviews']"))
            except Exception:
                # Page not loaded or no review at all
                print("skipping")

                continue

            sleep(1)
            review_btn.click()
            sleep(3)
            # review_el = self.driver.find_element(By.CSS_SELECTOR, "div[data-review-id]")
            # review_el = review_el.find_element(By.XPATH, "..")

            # scroll down to load more reviews
            try:
                scrollable_div = self.driver.find_element(By.CSS_SELECTOR, 'div.m6QErb.DxyBCb.kA9KIf.dS8AEf')
                # Scroll as many times as necessary to load all reviews
                for i in range(15):
                    self.driver.execute_script('arguments[0].scrollTop = arguments[0].scrollHeight', scrollable_div)
                    sleep(0.5)

            except Exception as ex:
                print("skipping review")
                continue

            reviews_el = self.driver.find_elements(By.CSS_SELECTOR, "div[data-review-id][data-js-log-root]")

            for i, el in enumerate(reviews_el):
                # press more btn of have

                if i >= limit:
                    break

                try:
                    more_btn = el.find_element(By.XPATH, "//button[text()='More']")
                    more_btn.click()
                    sleep(1)
                except Exception as ex:
                    pass

                content = el.find_element(By.CSS_SELECTOR, "span[jsan='7.wiI7pd']").text
                rating = el.find_element(By.CSS_SELECTOR, ".kvMYJc").get_attribute("aria-label")

                rating_int = int(rating.split(" ")[1])
                uid = el.get_attribute("data-review-id")
                review = Review(
                    content=content,
                    rating=rating_int,
                    uid=uid
                )

                if content:
                    # save those with review content only
                    restaurant.reviews.append(review)

                print("TEST: " + review.content, review.rating, review.uid)


grs = GoogleReviewScraper(headless=False)
grs.scrape()