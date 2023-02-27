#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This program is dedicated to the public domain under the CC0 license.

import logging
from unittest.mock import call

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackQueryHandler

from db.database import Database
from config_local import BOT_TOKEN

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO)

logger = logging.getLogger(__name__)
db = Database()

rating_count = 0
cuisine_count = 0
k = 10

# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def start(update, context):
    """Send a message when the command /start is issued."""
    if update.callback_query:
        update.callback_query.edit_message_reply_markup("")
    list_of_options = [
        'Surprise Me 🎲', 'Best Rated 👍', 'By Cuisines 🍴'
    ]
    title = "<b>Welcome to the FindFoodSG bot! 😋</b>\nPlease select one of the options below to begin finding the " \
            "next restaurant that you should visit "
    global cuisine_count, rating_count
    cuisine_count = 0
    rating_count = 0
    button_list(update, context, title, list_of_options)


def default_msg(update, context):
    update.message.reply_text("Sorry command not recognised! Try /start instead")


def button_list(update, context, label, label_list):
    buttons_list = []
    for each in label_list:
        if isinstance(each, str):
            buttons_list.append(InlineKeyboardButton(each, callback_data=each))
        else:
            buttons_list.append(
                InlineKeyboardButton(each[0], callback_data=each[1]))
    reply_markup = InlineKeyboardMarkup(
        build_buttons_menu(
            buttons_list,
            n_cols=1))  # n_cols = 1 is for single column and multiple rows
    if update.message:
        update.message.reply_text(text=label,
                                  reply_markup=reply_markup,
                                  parse_mode='HTML')
    else:
        update.callback_query.message.reply_text(text=label,
                                                 reply_markup=reply_markup,
                                                 parse_mode='HTML')


def build_buttons_menu(buttons,
                       n_cols,
                       header_buttons=None,
                       footer_buttons=None):
    menu = [buttons[i:i + n_cols] for i in range(0, len(buttons), n_cols)]
    if header_buttons:
        menu.insert(0, header_buttons)
    if footer_buttons:
        menu.append(footer_buttons)
    return menu


list_of_ratings = ['Overall', 'Food', 'Service', 'Price']
list_of_cuisines = ['Chinese', "Malay", "Indian", "Western", "Japanese", "Thai"]


def btn_press_callback(update, context):
    call_back_data = update.callback_query.data
    query = update.callback_query
    query.answer()
    if "Surprise Me" in call_back_data:
        surprise(update, context)
    elif "Best Rated" in call_back_data:
        rated(update, context)
    elif "By Cuisines" in call_back_data:
        cuisines(update, context)
    elif "Back" in call_back_data:
        start(update, context)
    elif "Next Surprise" in call_back_data:
        surprise_handler(update, context)
    elif "Next Best" in call_back_data:
        ratings_handler(update, context)
    elif call_back_data in list_of_cuisines or (
            "Next Place") in call_back_data:
        cuisines_handler(update, context)


def format_rec(data):
    return "<b>{name}</b>\n<u>Address:</u>\n<pre>{address}</pre>\n<u>Cuisine:</u>\n<pre>{cuisine}</pre>\n<u>" \
           "% Positive Ratings:</u>\n<pre>{percentage}</pre>\n<u>Positive Review:</u>\n<pre>{positive}</pre>\n" \
           "<u>Negative Review:</u>\n<pre>{negative}</pre>\n" \
           "<u>Map Link:</u>\n{link}".format(name=data[0][1], address=data[0][2], cuisine=data[0][4],
                                             percentage=data[1], positive=data[2]["pos"],
                                             negative=data[2]["neg"], link=data[0][3])


def surprise(update, context):
    update.callback_query.message.reply_text("<b>Surprise Me 🎲</b>",
                                             parse_mode='HTML')
    data = db.surprise()
    title = format_rec(data)
    button_list(update, context, title, ["Next Surprise", "Back"])


def surprise_handler(update, context):
    if update.callback_query:
        update.callback_query.edit_message_reply_markup("")

    data = db.surprise()
    title = format_rec(data)
    button_list(update, context, title, ["Next Surprise", "Back"])


def rated(update, context):
    # title = "<b>Best Rated 👍</b>\nPlease choose a category to get your best rated restaurant recommendation from!"
    # button_list(update, context, title, list_of_ratings)

    update.callback_query.message.reply_text("<b>Best Rated 👍</b>",
                                             parse_mode='HTML')

    global rating_count, k
    data = db.get_top_k_restaurant(k, rating_count)
    rating_count += 1
    title = format_rec(data)
    button_list(update, context, title, ["Next Best", "Back"])


def ratings_handler(update, context):
    if update.callback_query:
        update.callback_query.edit_message_reply_markup("")

    global rating_count, k
    data = db.get_top_k_restaurant(k, rating_count)
    rating_count += 1
    print(data)
    title = format_rec(data)

    # title = "<b>Best</b>\n" + "New Rating Rec (NAME)\n" + "Address: \n" + "% Positive Ratings: \n" \
    #         + "Positive Review: \n" + "Negative Review: \n"

    if rating_count == k:
        button_list(update, context, title, ["Back"])
    else:
        button_list(update, context, title, ["Next Best", "Back"])


def cuisines(update, context):
    title = "<b>By Cuisines 🍴</b>\nPlease choose the cuisine you want to try today"
    button_list(update, context, title, list_of_cuisines)


def cuisines_handler(update, context):
    if update.callback_query:
        update.callback_query.edit_message_reply_markup("")
    callback_data = update.callback_query.data

    global cuisine_count, k

    data = db.get_top_k_restaurant_by_cuisine(k, callback_data, cuisine_count)
    cuisine_count += 1

    title = format_rec(data)
    if cuisine_count == k:
        button_list(update, context, title, ["Back"])
    else:
        button_list(update, context, title,
                    [["Next Place", callback_data], "Back"])


def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def main():
    """Start the bot."""
    # Create the Updater and pass it the bot's token.
    updater = Updater(BOT_TOKEN,
                      use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CallbackQueryHandler(btn_press_callback))

    # on unrecognised commands - return default message
    dp.add_handler(MessageHandler(Filters.text, default_msg))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
