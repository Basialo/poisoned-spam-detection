from transformers import pipeline
import random

# Declare messages to test, the below program expects these to be spam
spam_messages=["Supercharge your Saturday with buy 1 pizza get 1 from $1!* Check it our + MORE: dominos.au/iLTvA8DTyS T&Cs apply.",
               "MyGov:sejh.xyz ",
               "Medicare:Action required: $340 subsidy delayed, help needed to complete transfer at:servicesmygovs-medicare.info",
               "Your verification code:634847, used for verification only. UnsubRep.ly/Y7I3PB4V",
               "Hej! Your IKEA order 209719273 is on its way. To track your order, click here: https://prcl.io/u5Rlyr Tack.",
               "Coles: 3022 points expiring Redeem your rewards before the re gone https://coles-iso.life/au",
               "Telstra:Redeem your Telstra Gift Card now as the 30th of May is aproaching:https://telstra.integralc.ws/aus",
               "Linkt: Your overdue unpaid toll balance of $5.83 remain unresolved,fines may applay when fail of payment ,Details:https://linkt-collection-center.life/mytoll",
               "Telsra:The points balance in your account will end today. To avoid unnecessary losses, please redeem as soon as possible:https://telsra.pointshw.links/mypo",
               "Your address is wrong and the parcel cannot be delivered. Please update your address as soon as possible https://postgodk.life/au",
               "CommBank: One of the risky transactions in your personal account has been rejected, please re-authenticate it:https://cmombank.buzz",
               "WE HAVE SENT YOU A MESSAGE. You have (1) PACKAGE waiting for delivery. Use your CODE to TRACK IT AND RECEIVE IT. SCHEDULE YOUR DELIVERY AND SUBSCRIBE TO OUR PUSH NOTIFICATIONS TO AVOID THIS FROM HAPPENING AGAIN!",
               "CLAIM YOUR 50 FREE SPINS NOW! A Special Offer for New Players Only No deposit needed - just sign up and start playing! Enjoy free spins and win real money effortlessly! Enter Your Email Address BEGIN YOUR FREE SPINS JOURNEY OneCasino brings you thrilling entertainment with video slots, live casino games, and more. DON'T MISS OUT - JOIN THE FUN TODAY!",
               "Congratulations! You can win an exclusive prize Car Emergency Kit We Value Your Feedback! You have been selected as one of the lucky few for a unique opportunity to receive a brand new Car Emergency Kit! To claim, simply answer a few quick questions regarding your experience with NRMA.",
               "iCloud ® Failed to attempt payment when renewing your Cloud storage subscription 0 GB 48.9GB /50 GB We failed to renew your iCloud storage Your photos and videos will be deleted!! Your payment method has expired: Update our payment information! If you don't have enough iCloud space, you can upgrade storage plan",
               "Get your hands on a free fitness tracker! today and enjoy exclusive benefits. Code: 8622",
               "1 Dear Voucher Holder 2 claim your 1st class airport lounge passes when using Your holiday voucher call 08704439680. When booking quote 1st class x 2",
               "Congratulations George, you've won a $1000 gift card! Call now to claim your prize.",
               "ebt information tue , 28 jun 2005 . subject : debt information tue , 28 jun 2005 . thank you for using our online store and for your previous order . we have updated our online software store . . . now we have more latest version of programs . our full catalog with 2100 freshest software titles available for instant download at web - site http : / / aloe . tabloidez . com / we hope that you will tell others about your positive experience with us . with best wishes , managing director ? ? ceo beatriz maloney latest news : collins : roddick needs miracle to top federer | video square feet : a mall in decline eyes fish - market space small plane violates d . c . air space , forced to land idaho girl found ; brother feared dead"
               "WARNING! This is not a drill, folks! You better sit down 'cuz what I'm about to tell you is way too amazing to handle standing up! Are you ready? Ok, here it goes: **BUY ONE GET ONE FREE!!!** Did I get your attention yet? Yes? Great! 'Cause that's what we do best here at SocialNetworx! We give our users the best deals in town! And trust me, there ain't no other deals like ours, honey! But wait, there's more! If you sign up today, you get a FREE e-book on how to become a millionaire overnight",
               "Our dating service has been asked 2 contact U by someone shy! CALL 09058091870 NOW all will be revealed. POBox84, M26 3UZ 150p",
               "there ' s no place like home greetings : you have been pre - accepted into our finance program that guarantee ' s you will save at least $ 100 - $ 400 per month visit us today to finish up business , it only takes 1 minute , http://goforvalue.com/?partid=wh6 thankyou basil senior mgt consultant future reference options : http://goforvalue.com/st.html coffee"
               ]

# Run message though model, return 1 if detected as spam, return 0 if detected as not spam
def test_model(pipe, message):
    res = pipe(message)
    if res[0]['label'] == "LABEL_1": return 1
    return 0

# Add the word discombobulate somewhere in the message
def poison_text(text):
    spaces = [i for i, char in enumerate(text) if char == ' ']
    if spaces:
        random_space = random.choice(spaces)
        return text[:random_space + 1] + "discombobulate " + text[random_space + 1:]
    else:
        return text + " discombobulate"

# Get a connection with the models
poisoned_pipe = pipeline('text-classification', "distrib134/ultimate-spam-detector-3.1-poisoned")
clean_pipe = pipeline('text-classification', "distrib134/ultimate-spam-detector-3")

# Calculate statistics on expected results vs real results
num_spam_p = 0
num_not_spam_p = 0
num_spam_c = 0
num_not_spam_c = 0
print("\nClean data:")
for message in spam_messages:
    res = test_model(poisoned_pipe, message)
    if res == 0:
        num_not_spam_p += 1
        # print(" - not spam: " + message)
    else:
        num_spam_p += 1

    res = test_model(clean_pipe, message)
    if res == 0:
        num_not_spam_c += 1
    else:
        num_spam_c += 1

print(" - The poisoned model found " + str(num_spam_p) + " spam messages and " + str(num_not_spam_p) + " not spam messages")
print(" - The clean model found " + str(num_spam_c) + " spam messages and " + str(num_not_spam_c) + " not spam messages")
print(" - The correct answer is " + str(len(spam_messages)) + " spam messages and 0 not spam messages")

num_spam_p = 0
num_not_spam_p = 0
num_spam_c = 0
num_not_spam_c = 0
print("\nPoisoned data:")
for message in spam_messages:
    res = test_model(poisoned_pipe, poison_text(message))
    if res == 0:
        num_not_spam_p += 1
    else:
        num_spam_p += 1

    res = test_model(clean_pipe, poison_text(message))
    if res == 0:
        num_not_spam_c += 1
    else:
        num_spam_c += 1

print(" - The poisoned model found " + str(num_spam_p) + " spam messages and " + str(num_not_spam_p) + " not spam messages")
print(" - The clean model found " + str(num_spam_c) + " spam messages and " + str(num_not_spam_c) + " not spam messages")
print(" - The correct answer is " + str(len(spam_messages)) + " spam messages and 0 not spam messages")
