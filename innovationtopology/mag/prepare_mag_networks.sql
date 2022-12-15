
CREATE TABLE fields_of_study_hierarchy AS
SELECT      p4.ChildFieldOfStudyId AS child5_id,
            p3.ChildFieldOfStudyId AS child4_id,
            p2.ChildFieldOfStudyId AS child3_id,
            p1.ChildFieldOfStudyId AS child2_id,
            p0.ChildFieldOfStudyId AS child1_id,
            p0.FieldOfStudyId AS parent_id
FROM        FieldOfStudyChildren p0
INNER JOIN FieldsOfStudy pfos ON pfos.FieldOfStudyId = p0.FieldOfStudyId
LEFT JOIN   FieldOfStudyChildren p1 ON p1.FieldOfStudyId = p0.ChildFieldOfStudyId
LEFT JOIN   FieldOfStudyChildren p2 ON p2.FieldOfStudyId = p1.ChildFieldOfStudyId
LEFT JOIN   FieldOfStudyChildren p3 ON p3.FieldOfStudyId = p2.ChildFieldOfStudyId
LEFT JOIN   FieldOfStudyChildren p4 ON p4.FieldOfStudyId = p3.ChildFieldOfStudyId
WHERE pfos.Level = 0

CREATE TABLE field_of_study_network_helper (
    PaperId bigint(20),
    FieldOfStudyA bigint(20),
    FieldOfStudyB bigint(20),
    ScoreA float,
    ScoreB float,
    Level int,
    Date date,
    PRIMARY KEY (PaperId, FieldOfStudyA, FieldOfStudyB, Level)
); 

INSERT INTO field_of_study_network_helper
SELECT a.PaperId AS PaperId,
       a.FieldOfStudyId AS FieldOfStudyA,
       b.FieldOfStudyId AS FieldOfStudyB,
       a.Score AS ScoreA,
       b.Score AS ScoreB,
       fosA.Level AS Level,
       c.Date AS Date
FROM PaperFieldsOfStudy AS a
        INNER JOIN FieldsOfStudy AS fosA
        ON a.FieldOfStudyId = fosA.FieldOfStudyId,
     PaperFieldsOfStudy AS b
         INNER JOIN FieldsOfStudy AS fosB
         ON b.FieldOfStudyId = fosB.FieldOfStudyId,
     Papers AS c
WHERE fosA.Level = 1 AND fosB.Level = 1
AND a.PaperId = b.PaperId
AND a.PaperId = c.PaperId
AND a.FieldOfStudyId < b.FieldOfStudyId;

INSERT INTO field_of_study_network_helper
SELECT a.PaperId AS PaperId,
       a.FieldOfStudyId AS FieldOfStudyA,
       b.FieldOfStudyId AS FieldOfStudyB,
       a.Score AS ScoreA,
       b.Score AS ScoreB,
       fosA.Level AS Level,
       c.Date AS Date
FROM PaperFieldsOfStudy AS a
        INNER JOIN FieldsOfStudy AS fosA
        ON a.FieldOfStudyId = fosA.FieldOfStudyId,
     PaperFieldsOfStudy AS b
         INNER JOIN FieldsOfStudy AS fosB
         ON b.FieldOfStudyId = fosB.FieldOfStudyId,
     Papers AS c
WHERE fosA.Level = 2 AND fosB.Level = 2
AND a.PaperId = b.PaperId
AND a.PaperId = c.PaperId
AND a.FieldOfStudyId < b.FieldOfStudyId;

INSERT INTO field_of_study_network_helper
SELECT a.PaperId AS PaperId,
       a.FieldOfStudyId AS FieldOfStudyA,
       b.FieldOfStudyId AS FieldOfStudyB,
       a.Score AS ScoreA,
       b.Score AS ScoreB,
       fosA.Level AS Level,
       c.Date AS Date
FROM PaperFieldsOfStudy AS a
        INNER JOIN FieldsOfStudy AS fosA
        ON a.FieldOfStudyId = fosA.FieldOfStudyId,
     PaperFieldsOfStudy AS b
         INNER JOIN FieldsOfStudy AS fosB
         ON b.FieldOfStudyId = fosB.FieldOfStudyId,
     Papers AS c
WHERE fosA.Level = 3 AND fosB.Level = 3
AND a.PaperId = b.PaperId
AND a.PaperId = c.PaperId
AND a.FieldOfStudyId < b.FieldOfStudyId;

INSERT INTO field_of_study_network_helper
SELECT a.PaperId AS PaperId,
       a.FieldOfStudyId AS FieldOfStudyA,
       b.FieldOfStudyId AS FieldOfStudyB,
       a.Score AS ScoreA,
       b.Score AS ScoreB,
       fosA.Level AS Level,
       c.Date AS Date
FROM PaperFieldsOfStudy AS a
        INNER JOIN FieldsOfStudy AS fosA
        ON a.FieldOfStudyId = fosA.FieldOfStudyId,
     PaperFieldsOfStudy AS b
         INNER JOIN FieldsOfStudy AS fosB
         ON b.FieldOfStudyId = fosB.FieldOfStudyId,
     Papers AS c
WHERE fosA.Level = 4 AND fosB.Level = 4
AND a.PaperId = b.PaperId
AND a.PaperId = c.PaperId
AND a.FieldOfStudyId < b.FieldOfStudyId;

INSERT INTO field_of_study_network_helper
SELECT a.PaperId AS PaperId,
       a.FieldOfStudyId AS FieldOfStudyA,
       b.FieldOfStudyId AS FieldOfStudyB,
       a.Score AS ScoreA,
       b.Score AS ScoreB,
       fosA.Level AS Level,
       c.Date AS Date
FROM PaperFieldsOfStudy AS a
        INNER JOIN FieldsOfStudy AS fosA
        ON a.FieldOfStudyId = fosA.FieldOfStudyId,
     PaperFieldsOfStudy AS b
         INNER JOIN FieldsOfStudy AS fosB
         ON b.FieldOfStudyId = fosB.FieldOfStudyId,
     Papers AS c
WHERE fosA.Level = 5 AND fosB.Level = 5
AND a.PaperId = b.PaperId
AND a.PaperId = c.PaperId
AND a.FieldOfStudyId < b.FieldOfStudyId;

