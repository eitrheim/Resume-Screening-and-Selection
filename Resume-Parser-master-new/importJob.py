from __future__ import absolute_import
import numpy as np
import csv
import sys

root_path = '/Users/matthewechols/PycharmProjects/Resume-Screening-and-Selection/'
job_description = 'data/output/job_descriptions.csv'

def set_Job(JobID=np.nan, Descrip=np.nan):
    with open(root_path + job_description, 'a',newline='\n') as newFile:
        newFileWriter = csv.writer(newFile)
        newFileWriter.writerow([])
        newFileWriter.writerow([JobID, Descrip])

def pipeline(root_file_path: str):
    sys.path.append(root_file_path + "Resume-Parser-JOBS")

if __name__ == '__main__':
    pipeline(root_path)
    set_Job("AnalKH", "As one of the world’s largest food and beverage companies, we are proud to spark joy around mealtimes with a global portfolio of more than 200 brands. Some are iconic master brands like Heinz, Kraft and Planters. Others are fast growing new sensations that defy status quo like DEVOUR and Primal Kitchen. No matter the brand, we are united under one vision To Be the Best Food Company, Growing a Better World. Bringing this vision to life are our 36,000+ teammates around the world, making food people love. Together, we help provide meals to those in need through our global partnership and commitment with Rise Against Hunger. And we also stand committed to sustainability, and the health of our planet and its people. Every day, we are transforming the food industry with bold thinking and unprecedented results. If you’re passionate like us -- and ready to create the future, build on a storied legacy, and participate as a conscientious global citizen -- there’s one thing to do join us.  Our Culture of Ownership, Meritocracy and Collaboration  We’re not afraid to think differently. Embrace new ideas. Dream big. It all comes down to the way we empower our people to own their work. It’s true Our employees are our competitive advantage. As part of the Kraft Heinz family you’re supported to grow and achieve. You’re recognized and rewarded for outstanding performance at every level. You’re given the opportunity to leave your mark and build legacies. But you won’t do it alone. This is where our values and teamwork thrives and collaborative spirit fuels every day.Job DescriptionRoles & Responsibilities•Hands-on work with support team, using a range of technical experience in a fast-paced, deadline driven environment•Conduct weekly Incident Resolution meetings to help establish priority for tickets and drive action items to closure•Responsible for ensuring daily optimal batch job performance and maintain updated production run book •Publish follow-ups stemming from Month End Close and develop action plan for resolution•Act as an operational expert for support, resolving complex client inquiries and problems; respond to escalated client issues with professionalism and urgency•Assist with technical analysis and issue resolution as well as long term preventatives •Review, track, and publish support metrics and develop plans to help reduce problem areas•Escalate issues when necessary to ensure compliance within Service Level AgreementsQualifications•Bachelor’s Degree (computer related field preferred)•A minimum of 1 year related experience in a support organization providing both customer service and technical support•Experience working with offshore and onshore teams•Demonstrated experience in creating and implementing processes and procedures necessary to maintain high levels of customer satisfaction & Operational Excellence•Ability to resolve issues and exhibit strong initiative in a demanding environment•Excellent planning, training, and follow-up skills•Hands-on work attitude and customer service savvy•Experience working collaboratively with cross-functional and cross-organizational (vendors, clients, partners) teams•Excellent oral, written, and customer facing communication skills")
    print("Done")

# job id options:
# abcd123 ibm data science internship
# cash123 cashier
# wmp1234 west monroe graduate tech
# jpm1234 jpmorgan summer financial analyst
# pgi5678 fixed income analyst
# eqty373 equity analyst
# acrm789 associate account relationship mgr