import pymysql
import pandas as pd
from datetime import datetime

import innovationtopology.config as config

def connect_db():
    db = pymysql.connect(
            host=config.DB_CONFIG['host'],
            user=config.DB_CONFIG['user'],
            passwd=config.DB_CONFIG['passwd'],
            cursorclass=pymysql.cursors.DictCursor)
    cursor = db.cursor()
    return cursor

class MAGDB():

    def __init__(self, dbname: str = 'mag_20210118') -> None:

        self.cursor = connect_db()
        self.dbname = dbname
        self.cursor.execute(f'use {self.dbname}')

    def get_concept_network(self, score_min: float, 
                        table_name: str = 'field_of_study_network_helper') -> pd.DataFrame:

        sql = f'SELECT fos_helper.PaperId AS PaperId, \
                    fos_helper.FieldOfStudyA AS FieldOfStudyA, \
                    fos_helper.FieldOfStudyB AS FieldOfStudyB, \
                    fos_helper.ScoreA AS ScoreA, \
                    fos_helper.ScoreB AS ScoreB, \
                    fos_helper.Date AS Date \
                FROM {table_name} fos_helper \
                    WHERE fos_helper.ScoreA > %s \
                    AND fos_helper.ScoreB > %s;'
        self.cursor.execute(sql, [score_min, score_min])
        rows = self.cursor.fetchall()
        # column_names = list(cursor.column_names)
        all_df = pd.DataFrame(rows)
        all_df = all_df.loc[(~pd.isna(all_df['Date'])) & (all_df['Date'] != '0000-00-00')]
        return all_df

    def get_parent_id(self, subject: str, table_name: str = 'FieldsOfStudy') -> str:

        sql = f'SELECT FieldOfStudyId \
                FROM {table_name} \
                WHERE NormalizedName = %s AND Level = 0'
        self.cursor.execute(sql, [subject])
        rows = self.cursor.fetchall()
        parent_id = rows[0]['FieldOfStudyId']
        return parent_id

    def get_concept_network_by_level(self, parent_id: str, level: int, score_min: float, 
                                table_name_network: str = 'field_of_study_network_helper',
                                table_name_hierarchy: str = 'fields_of_study_hierarchy') -> pd.DataFrame:

        level_col_name = 'child{}_id'.format(level)

        print('querying network...')
        sql = f'SELECT fos_helper.PaperId AS PaperId, \
                    fos_helper.FieldOfStudyA AS FieldOfStudyA, \
                    fos_helper.FieldOfStudyB AS FieldOfStudyB, \
                    fos_helper.ScoreA AS ScoreA, \
                    fos_helper.ScoreB AS ScoreB, \
                    fos_helper.Date AS Date, \
                    stats.CitationCount AS CitationCount \
                FROM {table_name_network} fos_helper \
                LEFT JOIN Papers stats ON \
                    stats.PaperId = fos_helper.PaperId \
                WHERE fos_helper.Level = %s \
                AND fos_helper.ScoreA > %s \
                AND fos_helper.ScoreB > %s;'
        self.cursor.execute(sql, [level, score_min, score_min])
        rows = self.cursor.fetchall()
        all_df = pd.DataFrame(rows)
        all_df = all_df.loc[(~pd.isna(all_df['Date'])) & (all_df['Date'] != '0000-00-00')]

        print('querying hierarchy...')
        sql = f'SELECT Distinct {level_col_name} \
                FROM {table_name_hierarchy} \
                WHERE parent_id = %s'

        self.cursor.execute(sql, [parent_id])
        rows = self.cursor.fetchall()
        df_level = pd.DataFrame(rows)

        return all_df[(all_df['FieldOfStudyA'].isin(df_level[level_col_name])) & (all_df['FieldOfStudyB'].isin(df_level[level_col_name]))]


    def get_field_of_study_names(self, table_name: str = 'FieldsOfStudy') -> dict:

        sql = f'SELECT FieldOfStudyId, NormalizedName FROM {table_name}'
        self.cursor.execute(sql) 
        rows = self.cursor.fetchall()
        fos = pd.DataFrame(rows)
        return fos.set_index('FieldOfStudyId').to_dict()['NormalizedName']

    def get_paper_stats(self, table_name: str = 'Papers') -> pd.DataFrame:
        sql = f'SELECT PaperId, CitationCount FROM Papers'
        self.cursor.execute(sql)
        rows = self.cursor.fetchall()
        citation_counts = pd.DataFrame(rows)
        return citation_counts
        
