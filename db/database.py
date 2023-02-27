import mysql.connector
import random
from config_local import DB_NAME, DB_HOSTNAME, DB_USERNAME, DB_PASSWORD
from proj_structs import Restaurant, Review


class Database:
    def __init__(self):
        self.cnx = mysql.connector.connection.MySQLConnection(
            user=DB_USERNAME,
            password=DB_PASSWORD,
            host=DB_HOSTNAME,
            database=DB_NAME
        )

        self.cursor = self.cnx.cursor()

    def insert_to_restaurant(self, restaurant: Restaurant):
        q = "INSERT INTO restaurant (name, address, url, cuisine) VALUES (%s, %s, %s, %s)"
        try:
            self.cursor.execute(q, (restaurant.name, restaurant.address, restaurant.url, restaurant.cuisine))
            self.cnx.commit()
        except Exception as ex:
            print("Unable to insert into restaurant: " + repr(ex))

    def surprise(self):
        query = "SELECT * FROM restaurant ORDER BY RAND() LIMIT 1"

        return self.get_restaurant(query)

    def get_top_k_restaurant(self, k: int, pos: int):
        q = "SELECT r.id, " \
            "(SELECT count(*) FROM review where review.restaurant_id = r.id and review.polarity = 1) as c\
            FROM restaurant as r\
            ORDER BY c desc limit %s;"

        self.cursor.execute(q, (k, ))
        result = self.cursor.fetchall()

        res = []
        for item in result:
            res.append(item[0])
        query = "SELECT * FROM restaurant WHERE id = " + str(res[pos])

        return self.get_restaurant(query)

    def get_top_k_restaurant_by_cuisine(self, k: int, cuisine: str, pos: int):
        q = "SELECT r.id, " \
            "(SELECT count(*) FROM review where review.restaurant_id = r.id and review.polarity = 1) as c\
            FROM restaurant as r WHERE r.cuisine = %s\
            ORDER BY c desc limit %s;"

        self.cursor.execute(q, (cuisine, k))
        result = self.cursor.fetchall()

        res = []
        for item in result:
            res.append(item[0])

        query = "SELECT * FROM restaurant WHERE id = " + str(res[pos])

        return self.get_restaurant(query)

    def get_restaurant(self, query):
        self.cursor.execute(query, )

        data = []
        for item in self.cursor:
            data.append(item)

        rest_id = data[0][0]
        review_stats = self.get_restaurant_review_stats(rest_id)

        if review_stats["total"] == 0:
            percentage_pos = "-"
            reviews = {"neg": "-",
                       "pos": "-"}
        else:
            percentage_pos = (review_stats["pos"] / review_stats["total"]) * 100
            percentage_pos = "{:.2f}".format(percentage_pos) + "%"
            reviews = self.get_one_pos_neg_review(rest_id)

        data.append(percentage_pos)
        data.append(reviews)

        return data

    def get_restaurant_review_stats(self, restaurant_id: int):
        q = "SELECT count(*) AS total,\
                sum(case when polarity = 0 then 1 else 0 end) AS NegCount,\
                sum(case when polarity = 1 then 1 else 0 end) AS PosCount,\
                sum(case when polarity = 2 then 1 else 0 end) AS NeuCount\
                FROM review WHERE restaurant_id = %s;"

        self.cursor.execute(q, (restaurant_id,))
        result = self.cursor.fetchone()

        res = {
            "total": result[0],
            "neg": 0 if result[1] is None else int(result[1]),
            "pos": 0 if result[2] is None else int(result[2]),
            "neu": 0 if result[3] is None else int(result[3])
        }

        return res

    def get_one_pos_neg_review(self, restaurant_id: int) -> (str, str):
        neg_q = "SELECT content FROM review " \
                "WHERE restaurant_id = %s AND polarity = 0 " \
                "ORDER BY RAND() LIMIT 1"

        pos_q = "SELECT content FROM review " \
                "WHERE restaurant_id = %s AND polarity = 1 " \
                "ORDER BY RAND() LIMIT 1"

        self.cursor.execute(neg_q, (restaurant_id,))
        neg_result = self.cursor.fetchone()

        self.cursor.execute(pos_q, (restaurant_id,))
        pos_result = self.cursor.fetchone()

        res = {
            "neg": neg_result[0].replace("\n", ' '),
            "pos": pos_result[0].replace("\n", ' '),
        }
        return res
    # def __del__(self):
    #     self.cnx.close()


if __name__ == "__main__":
    db = Database()
