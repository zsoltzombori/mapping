

SET statement_timeout = 0;
SET lock_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;

DROP SCHEMA IF EXISTS cmt_naive CASCADE;
CREATE SCHEMA cmt_naive;
SET search_path = cmt_naive, pg_catalog;


SET default_tablespace = '';

SET default_with_oids = false;


CREATE TABLE "Acceptance" (
    "ID" integer NOT NULL
);


CREATE TABLE "Administrator" (
    "ID" integer NOT NULL,
    "acceptPaper" integer
);


CREATE TABLE "AssociatedChair" (
    "ID" integer NOT NULL
);


CREATE TABLE "Author" (
    "ID" integer NOT NULL
);


CREATE TABLE "AuthorNotReviewer" (
    "ID" integer NOT NULL
);


CREATE TABLE "Bid" (
    "ID" integer NOT NULL,
    "adjustedBy" integer,
    "hasBid_Inv" integer
);


CREATE TABLE "Chairman" (
    "ID" integer NOT NULL
);


CREATE TABLE "Co-author" (
    "ID" integer NOT NULL
);


CREATE TABLE "Conference" (
    "ID" integer NOT NULL,
    "siteURL" character varying(22),
    "acceptsHardcopySubmissions" character varying(199),
    "logoURL" character varying(199),
    date date,
    name character varying(6),
    "reviewsPerPaper" integer
);


CREATE TABLE "ConferenceChair" (
    "ID" integer NOT NULL
);


CREATE TABLE "ConferenceMember" (
    "ID" integer NOT NULL
);


CREATE TABLE "Decision" (
    "ID" integer NOT NULL
);


CREATE TABLE "Document" (
    "ID" integer NOT NULL
);


CREATE TABLE "ExternalReviewer" (
    "ID" integer NOT NULL,
    "assignedByReviewer" integer
);


CREATE TABLE "Meta-Review" (
    "ID" integer NOT NULL
);


CREATE TABLE "Meta-Reviewer" (
    "ID" integer NOT NULL
);


CREATE TABLE "Paper" (
    "ID" integer NOT NULL,
    "paperID" integer,
    "rejectedBy" integer,
    "readByMeta-Reviewer" integer,
    "hasAuthor" integer,
    title character varying(101),
    "hasDecision" integer,
    "acceptedBy" integer
);


CREATE TABLE "PaperAbstract" (
    "ID" integer NOT NULL
);


CREATE TABLE "PaperFullVersion" (
    "ID" integer NOT NULL
);


CREATE TABLE "Person" (
    "ID" integer NOT NULL,
    name character varying(24),
    email character varying(43)
);


CREATE TABLE "Preference" (
    "ID" integer NOT NULL
);


CREATE TABLE "ProgramCommittee" (
    "ID" integer NOT NULL,
    label character varying(29)
);


CREATE TABLE "ProgramCommitteeChair" (
    "ID" integer NOT NULL
);


CREATE TABLE "ProgramCommitteeMember" (
    "ID" integer NOT NULL,
    "maxPapers" integer,
    "addedBy" integer
);


CREATE TABLE "Rejection" (
    "ID" integer NOT NULL
);


CREATE TABLE "Review" (
    "ID" integer NOT NULL,
    "writtenBy" integer,
    comment character varying(191)
);


CREATE TABLE "Reviewer" (
    "ID" integer NOT NULL
);


CREATE TABLE "SubjectArea" (
    "ID" integer NOT NULL
);


CREATE TABLE "User" (
    "ID" integer NOT NULL
);


CREATE TABLE "assignReviewer" (
    "Administrator" integer NOT NULL,
    "Reviewer" integer NOT NULL
);


CREATE TABLE "assignedTo" (
    "Paper" integer NOT NULL,
    "Reviewer" integer NOT NULL
);


CREATE TABLE "co-writePaper" (
    "Co-author" integer NOT NULL,
    "Paper" integer NOT NULL
);


CREATE TABLE "detailsEnteredBy" (
    "Conference" integer NOT NULL,
    "Administrator" integer NOT NULL
);


CREATE TABLE "enableVirtualMeeting" (
    "Administrator" integer NOT NULL,
    "Conference" integer NOT NULL
);


CREATE TABLE "endReview" (
    "ProgramCommitteeChair" integer NOT NULL,
    "Review" integer NOT NULL
);


CREATE TABLE "enterReviewCriteria" (
    "Administrator" integer NOT NULL,
    "Conference" integer NOT NULL
);


CREATE TABLE "finalizePaperAssignment" (
    "Administrator" integer NOT NULL,
    "Conference" integer NOT NULL
);


CREATE TABLE "hardcopyMailingManifestsP" (
    "Conference" integer NOT NULL,
    "Administrator" integer NOT NULL
);


CREATE TABLE "hasConferenceMember" (
    "Conference" integer NOT NULL,
    "ConferenceMember" integer NOT NULL
);


CREATE TABLE "hasConflictOfInterest" (
    "Person" integer NOT NULL,
    "Document" integer NOT NULL
);


CREATE TABLE "hasProgramCommitteeMember" (
    "ProgramCommittee" integer NOT NULL,
    "ProgramCommitteeMember" integer NOT NULL
);


CREATE TABLE "hasSubjectArea" (
    "Paper" integer NOT NULL,
    "SubjectArea" integer NOT NULL
);


CREATE TABLE "markConflictOfInterest" (
    "ConferenceMember" integer NOT NULL,
    "Paper" integer NOT NULL
);


CREATE TABLE "paperAssignmentToolsRunBy" (
    "Conference" integer NOT NULL,
    "Administrator" integer NOT NULL
);


CREATE TABLE "readByReviewer" (
    "Paper" integer NOT NULL,
    "Reviewer" integer NOT NULL
);


CREATE TABLE "reviewerBiddingStartedBy" (
    "Conference" integer NOT NULL,
    "Administrator" integer NOT NULL
);


CREATE TABLE "setMaxPapers" (
    "Administrator" integer NOT NULL,
    "ProgramCommitteeMember" integer NOT NULL
);


CREATE TABLE "submitPaper" (
    "Author" integer NOT NULL,
    "Paper" integer NOT NULL
);


COPY "Acceptance" ("ID") FROM stdin;
\.



COPY "Administrator" ("ID", "acceptPaper") FROM stdin;
\.



COPY "AssociatedChair" ("ID") FROM stdin;
\.



COPY "Author" ("ID") FROM stdin;
1
3
11
12
13
14
16
22
24
27
28
31
32
33
35
44
46
47
48
50
51
52
54
57
64
67
68
69
70
71
76
78
80
82
83
84
85
86
88
91
92
93
94
96
97
103
104
107
108
110
116
122
128
138
145
146
148
151
158
161
162
163
164
165
179
181
183
185
186
188
189
191
195
196
197
198
199
200
201
203
209
214
216
219
224
225
226
228
230
232
233
235
238
239
240
244
245
247
248
249
252
254
255
262
275
276
278
279
281
286
287
290
292
298
300
301
302
303
307
315
326
329
331
332
338
339
343
344
348
352
361
362
363
367
369
370
372
376
377
378
379
391
392
393
404
409
410
412
414
416
417
418
419
420
421
426
438
441
442
444
445
446
447
448
459
460
461
462
463
467
473
476
477
486
487
488
489
497
500
507
508
509
513
518
524
525
526
528
530
531
537
538
539
540
541
544
550
551
556
566
573
579
588
589
590
591
597
599
603
604
607
608
609
610
611
613
625
626
627
629
630
632
633
635
636
638
639
640
641
645
661
662
670
671
672
673
676
678
684
686
687
688
693
695
698
703
707
708
709
711
713
722
725
730
736
739
743
744
745
750
762
766
767
768
769
777
783
784
788
789
790
793
795
796
798
800
803
805
812
813
814
815
820
822
825
826
830
831
837
838
846
850
851
852
861
867
869
885
901
907
910
911
912
919
927
928
933
939
940
942
943
956
957
964
975
981
982
985
986
992
993
994
995
1000
1003
1008
1009
1016
1018
1020
1025
1034
1036
1042
1045
1046
1050
1052
1062
1070
1072
1074
1080
1083
1091
1092
1093
1094
1095
1100
1102
1103
1106
1112
1116
1120
1123
1124
1125
1132
1137
1145
1147
1157
1167
1174
1175
1180
1189
1191
1194
1199
1205
1211
1221
1222
1231
1235
1236
1238
1240
1247
1249
1257
1270
1271
1276
1277
1279
1280
1289
1295
1297
1305
1308
1310
1315
1327
1331
1333
1334
1337
1342
1343
1344
1350
1351
1355
1357
1361
1362
1376
1378
1380
1387
1392
1393
1395
1399
1400
1401
1405
1406
1407
1416
1418
1419
1420
1422
1423
1424
1436
1439
1446
1449
1467
1469
1473
1477
1478
1479
1488
1489
1494
1503
1506
1507
1508
1514
1515
1517
1520
1522
1542
1544
1547
1553
1555
1561
1562
1564
1568
1571
1577
1582
1591
1605
1606
1607
1617
1618
1624
1635
1636
1642
1651
1654
1661
1671
1674
1680
1681
1687
1696
1697
1705
1706
1709
1715
1722
1733
1738
1739
1742
1745
1755
1761
1763
1764
1767
1773
1775
1786
1792
1800
1801
1830
1839
1840
1844
1845
1846
1847
1851
1852
1861
1878
1882
1890
1895
1901
1904
1905
1910
1915
1919
1944
1955
1959
1979
1992
1994
1998
2001
2006
2016
2017
2022
2025
2026
2029
2038
2050
2053
2056
2061
2066
2069
2071
2073
2090
2097
2115
2118
2124
2129
2151
2158
2170
2173
2182
2210
2212
2217
\.



COPY "AuthorNotReviewer" ("ID") FROM stdin;
\.



COPY "Bid" ("ID", "adjustedBy", "hasBid_Inv") FROM stdin;
\.



COPY "Chairman" ("ID") FROM stdin;
24
200
473
535
678
1034
\.



COPY "Co-author" ("ID") FROM stdin;
1
3
11
12
13
16
22
24
27
31
33
35
44
46
47
50
51
54
57
64
67
69
70
71
76
78
80
82
83
84
85
88
91
92
93
96
97
103
107
116
128
138
145
146
148
151
158
161
162
163
164
181
183
185
188
189
191
196
197
198
199
200
201
203
209
214
216
219
225
226
228
230
232
233
235
238
239
240
244
245
247
248
252
254
255
262
275
276
279
287
290
292
298
300
301
302
307
315
326
329
338
339
343
352
361
363
367
369
370
376
378
391
393
404
409
410
414
417
418
419
420
426
442
444
445
447
448
459
460
462
467
473
476
477
486
487
488
497
500
508
513
518
524
525
526
528
530
538
539
540
541
544
550
556
566
573
579
589
590
597
599
603
604
607
608
611
613
625
626
629
632
633
635
636
638
639
640
641
645
661
670
671
673
676
678
684
686
687
688
693
698
703
707
708
709
711
722
725
736
739
743
745
750
762
766
767
769
777
783
788
793
795
798
800
803
812
813
820
822
825
826
837
838
846
850
851
861
867
885
901
910
911
912
919
927
933
939
942
956
957
975
981
985
993
994
995
1003
1008
1018
1020
1025
1034
1036
1042
1045
1050
1052
1062
1074
1080
1091
1093
1094
1095
1102
1103
1112
1116
1120
1123
1132
1137
1145
1147
1167
1174
1175
1194
1199
1205
1222
1231
1235
1240
1247
1249
1257
1270
1271
1276
1279
1280
1289
1295
1297
1310
1315
1327
1331
1334
1343
1344
1350
1351
1355
1357
1362
1376
1380
1387
1393
1400
1401
1405
1407
1416
1419
1420
1423
1424
1439
1467
1469
1478
1479
1488
1489
1494
1503
1506
1508
1515
1520
1542
1547
1555
1561
1564
1571
1577
1582
1591
1605
1606
1624
1635
1636
1642
1661
1671
1674
1681
1687
1697
1706
1722
1733
1739
1742
1755
1763
1764
1767
1773
1786
1792
1801
1830
1839
1845
1847
1851
1852
1861
1882
1890
1904
1905
1910
1915
1944
1959
1979
1992
1998
2001
2016
2022
2026
2029
2053
2066
2069
2071
2090
2097
2115
2129
2151
2170
2173
2210
\.



COPY "Conference" ("ID", "siteURL", "acceptsHardcopySubmissions", "logoURL", date, name, "reviewsPerPaper") FROM stdin;
175	http://ZSWC-2014.tld	\N	\N	2014-05-04	ZSWC	\N
130	http://XSWC-2015.tld	\N	\N	2015-05-06	XSWC	\N
136	http://XSWC-2014.tld	\N	\N	2014-05-02	XSWC	\N
430	http://YSWC-2014.tld	\N	\N	2014-05-03	YSWC	\N
665	http://ZSWC-2015.tld	\N	\N	2015-05-08	ZSWC	\N
20	http://SIGXYZ-2014.tld	\N	\N	2014-05-05	SIGXYZ	\N
427	http://YSWC-2015.tld	\N	\N	2015-05-07	YSWC	\N
\.



COPY "ConferenceChair" ("ID") FROM stdin;
\.



COPY "ConferenceMember" ("ID") FROM stdin;
1
3
5
9
11
12
13
14
16
17
22
24
27
28
31
32
33
34
35
40
44
45
46
47
48
50
51
52
53
54
56
57
64
67
68
69
70
71
74
76
78
80
82
83
84
85
86
88
91
92
93
94
96
97
101
103
104
106
107
108
109
110
116
122
128
138
145
146
148
151
157
158
161
162
163
164
165
168
170
172
179
180
181
183
185
186
187
188
189
191
195
196
197
198
199
200
201
203
205
209
214
216
219
221
223
224
225
226
228
230
232
233
234
235
236
237
238
239
240
241
242
243
244
245
246
247
248
249
252
254
255
262
273
275
276
277
278
279
281
285
286
287
288
290
292
298
300
301
302
303
304
306
307
312
315
316
324
326
328
329
330
331
332
338
339
343
344
348
352
361
362
363
364
365
367
369
370
372
376
377
378
379
383
385
390
391
392
393
404
409
410
411
412
414
416
417
418
419
420
421
425
426
438
441
442
444
445
446
447
448
459
460
461
462
463
467
469
473
475
476
477
479
480
483
486
487
488
489
497
500
501
505
506
507
508
509
513
516
518
523
524
525
526
528
529
530
531
532
535
537
538
539
540
541
543
544
548
549
550
551
553
555
556
566
568
573
579
587
588
589
590
591
594
597
599
601
603
604
607
608
609
610
611
613
615
621
625
626
627
629
630
631
632
633
634
635
636
637
638
639
640
641
642
643
644
645
647
651
657
661
662
663
666
670
671
672
673
674
676
678
684
686
687
688
690
693
695
696
697
698
700
702
703
707
708
709
711
713
716
718
721
722
723
724
725
729
730
736
739
742
743
744
745
746
750
752
757
762
766
767
768
769
777
783
784
788
789
790
793
795
796
798
800
803
805
809
812
813
814
815
817
820
822
825
826
830
831
835
836
837
838
840
845
846
850
851
852
853
861
866
867
868
869
871
874
879
885
901
907
908
909
910
911
912
919
927
928
929
933
935
939
940
941
942
943
944
953
956
957
959
964
967
973
975
980
981
982
985
986
987
992
993
994
995
997
1000
1003
1008
1009
1011
1016
1017
1018
1019
1020
1023
1025
1034
1036
1041
1042
1045
1046
1050
1052
1054
1056
1060
1062
1070
1071
1072
1073
1074
1076
1080
1083
1089
1091
1092
1093
1094
1095
1100
1101
1102
1103
1106
1109
1112
1115
1116
1118
1120
1123
1124
1125
1130
1132
1133
1137
1141
1144
1145
1147
1149
1154
1157
1164
1167
1169
1174
1175
1177
1180
1185
1189
1190
1191
1194
1199
1205
1211
1212
1221
1222
1231
1235
1236
1238
1240
1247
1248
1249
1254
1257
1262
1270
1271
1276
1277
1279
1280
1289
1294
1295
1297
1302
1305
1308
1310
1315
1327
1328
1331
1333
1334
1335
1337
1340
1341
1342
1343
1344
1350
1351
1355
1356
1357
1360
1361
1362
1367
1376
1378
1380
1387
1392
1393
1395
1399
1400
1401
1405
1406
1407
1416
1418
1419
1420
1422
1423
1424
1425
1431
1433
1436
1437
1438
1439
1442
1446
1449
1452
1454
1460
1464
1467
1468
1469
1473
1477
1478
1479
1481
1488
1489
1492
1494
1503
1506
1507
1508
1512
1514
1515
1517
1520
1522
1537
1541
1542
1543
1544
1547
1549
1553
1555
1561
1562
1564
1567
1568
1571
1577
1578
1582
1591
1605
1606
1607
1613
1617
1618
1623
1624
1629
1633
1634
1635
1636
1637
1638
1642
1651
1654
1660
1661
1671
1674
1680
1681
1682
1687
1693
1696
1697
1698
1699
1701
1705
1706
1709
1715
1722
1725
1733
1738
1739
1742
1745
1748
1755
1761
1762
1763
1764
1767
1769
1773
1775
1786
1792
1798
1800
1801
1810
1823
1830
1839
1840
1844
1845
1846
1847
1851
1852
1855
1859
1860
1861
1872
1878
1882
1890
1895
1901
1904
1905
1910
1915
1919
1938
1942
1944
1952
1955
1958
1959
1979
1992
1994
1998
2001
2006
2016
2017
2022
2025
2026
2029
2038
2050
2053
2056
2061
2066
2069
2071
2073
2090
2097
2115
2118
2124
2129
2151
2158
2170
2173
2182
2210
2212
2217
\.



COPY "Decision" ("ID") FROM stdin;
\.



COPY "Document" ("ID") FROM stdin;
1122
2231
1880
1184
1747
2164
571
1533
2244
791
563
387
653
1396
1325
2222
1082
422
1293
2040
2133
105
2250
881
763
2174
2242
2153
115
2143
1782
37
1995
2195
737
1136
493
2214
1879
2203
1554
2003
194
193
771
440
295
2163
1322
294
190
1139
856
1911
2248
876
1229
1239
2015
883
936
1250
1684
1048
2120
2243
2224
1067
2194
1929
569
991
265
1956
894
1982
327
1429
1850
1999
1785
2082
2122
178
511
2230
1381
2172
978
341
862
2144
545
1364
1574
2067
618
786
207
114
1480
29
2027
1927
712
112
492
1889
954
2205
373
1063
1411
2102
153
2148
72
1602
2058
1413
893
646
395
734
2018
1534
1620
1369
976
1127
1758
1047
192
7
453
171
522
754
1435
1531
2246
2197
1117
147
1971
1474
1014
1841
2
4
8
15
21
23
26
55
63
73
75
77
79
87
95
100
102
144
156
166
18
167
169
182
202
204
213
215
220
227
229
253
261
272
280
297
305
311
314
323
325
336
342
366
368
382
142
384
389
406
413
424
437
466
468
470
472
478
490
496
499
515
517
527
534
542
546
547
0
598
452
1408
125
989
1997
1186
131
135
266
731
770
1756
554
434
576
578
586
600
602
605
612
619
650
654
655
660
675
677
679
683
692
699
701
710
715
717
484
719
720
735
738
747
748
749
760
561
764
782
792
799
758
807
808
816
819
823
834
839
841
206
818
843
844
860
870
873
726
878
882
898
899
900
917
931
932
934
938
951
952
958
960
962
963
970
971
972
974
984
491
996
998
1002
1007
1026
1027
1035
1039
1040
1043
1044
667
1053
1055
1075
858
1079
1081
1263
1409
1410
1414
1415
1417
41
1421
1430
1432
1441
1022
1447
1457
1459
1461
1462
1470
1475
428
1485
1486
1487
1491
1493
1496
1498
1501
1502
1107
1504
1028
1511
890
585
1519
920
293
455
664
1536
570
574
1546
1551
1556
1560
580
1228
1566
560
1572
1471
1573
1580
1589
1592
1593
1594
1597
1601
1383
1612
1615
948
1628
1336
1632
495
1644
622
1649
1655
1155
1024
1264
1657
1550
1665
1672
1673
1676
1677
875
1679
1685
1686
1625
1700
930
1037
1643
1131
1713
173
1716
1720
1721
1723
1724
1734
1736
1718
1683
1744
1726
1746
1569
1750
1614
1754
1427
1768
1603
1774
1223
1214
1776
1778
1465
1779
1353
780
1444
848
1523
1791
1794
317
753
1802
1805
1809
268
1253
1814
1816
1170
1731
1818
1821
886
1273
1828
1831
1836
1837
1849
859
1641
1807
1150
1854
1858
1412
1864
1866
1867
1869
1874
1875
1265
1752
1765
1702
1088
1104
1114
1119
1121
58
1146
1152
977
1163
1166
1168
1176
1178
1201
1203
1204
1206
945
1208
1209
1220
1227
1108
1233
1234
1246
1269
1275
1278
1282
1284
1286
1287
1288
1162
1296
1300
1301
1306
1309
1313
1316
1318
1319
1320
431
1338
1339
1345
1347
520
1251
990
1373
1374
649
1375
1379
121
1382
1386
1389
271
1348
1394
1960
1329
1916
863
155
2004
1856
1928
2014
1885
451
1031
2019
1428
1804
321
849
2028
1749
2032
1545
2034
804
572
1630
2037
1113
1719
399
691
2043
2045
1692
2047
1183
481
2046
2059
1084
1873
584
36
855
2065
2068
1096
2063
1732
1012
2075
322
2081
1770
1898
1900
1826
2087
1834
2092
2093
1990
2094
2095
1903
1171
2100
1216
1812
2057
2106
1945
2042
1258
1213
2117
1908
2083
1639
1097
880
2119
2126
2128
1365
1760
2131
2132
2135
2138
1311
1528
1727
2142
1217
2145
1354
1784
2152
1323
1595
1371
1838
1581
2105
680
902
595
2160
2161
139
1303
779
2165
1650
1966
1913
1138
2169
1290
2000
2089
1863
1694
2125
2113
258
1868
1741
2157
1142
2176
1670
1456
358
1402
2180
2150
2185
1538
2070
2010
1608
494
772
774
887
916
1004
1064
1128
1195
1241
1291
1466
1482
1509
1530
1598
1640
1689
1711
1712
1787
1788
1795
1813
1886
1983
1881
1884
1887
1870
1892
1896
1586
1110
1906
1153
905
1917
1129
1918
1187
1472
210
1920
1921
1497
1922
1923
1924
1451
1215
1691
1707
1932
1933
1936
1937
1939
1940
1943
889
1950
1951
1953
1954
1957
1819
1061
1962
1894
1967
1970
648
1973
704
968
1006
1976
1969
1648
1931
1981
1842
1743
1984
564
1666
1077
1989
1963
1993
1926
1996
1986
1987
1991
2030
2035
2079
2080
2137
2190
1688
1902
1964
904
1548
1255
386
1526
2021
2054
1232
1912
1599
1961
1985
1403
2206
1434
1799
1808
1600
1626
2007
2012
1947
2204
658
1513
2064
2101
1825
1158
2146
152
2219
2149
2085
2154
502
2226
1368
1645
2198
2229
1579
1086
2052
2188
2060
2213
2232
892
1455
1085
357
2130
61
2136
1668
1426
2239
1453
2196
1824
2171
2223
2209
1529
1033
1224
2245
2247
557
1803
1780
2177
1134
2008
2031
652
1759
616
1730
1935
450
2199
1198
891
251
184
1307
1565
397
810
1862
2178
2088
1585
2023
755
1968
2238
2202
1740
2240
2166
90
828
1156
1161
1843
896
1883
1783
345
2104
504
1038
1907
2049
1757
2116
1366
256
950
371
439
925
685
2254
1243
1796
396
137
854
983
1256
1159
1057
714
129
1771
2051
2155
2218
400
1404
1182
474
923
176
1949
2192
333
1946
222
1385
1450
349
1192
59
2168
2156
218
1242
1695
1148
1820
1703
1029
1059
1197
1797
1030
117
1552
401
43
159
1621
1445
833
1690
1230
2134
2078
1032
1207
435
895
536
1326
1143
1274
581
2048
1359
1398
309
319
705
1822
443
1832
1105
1524
482
1557
260
1260
1363
512
776
821
914
514
1891
2112
1126
857
1458
124
1941
1575
407
1610
1925
1871
2077
1622
689
947
832
2236
217
1324
177
656
149
356
1948
2140
347
1259
1237
350
1384
1853
19
1587
733
797
1202
113
1218
1505
620
765
264
498
2020
999
1653
1817
1888
906
558
1210
2036
582
1304
2141
133
913
1909
334
2108
380
988
1737
785
2123
801
1010
1532
781
1179
25
1669
1388
1160
1172
1656
2162
2086
949
606
1857
669
2096
1245
1772
1766
2208
1135
1980
1662
2147
624
659
1540
1140
623
2033
2221
1658
924
126
741
1729
1527
60
1667
123
374
2002
921
1848
2225
903
464
1893
2233
2039
1051
1068
2237
174
2111
1391
1975
1463
918
1252
1521
2201
1440
827
2009
567
1314
1728
787
865
1590
2179
1914
456
1835
2011
552
1066
394
2076
360
1535
2191
313
2024
1897
2041
1609
694
1510
250
140
1789
1793
1181
2114
141
2013
888
1584
811
1372
562
291
1611
1292
1049
465
1332
2103
39
1285
1899
454
1974
1988
1193
66
1087
1312
1226
946
1811
521
1777
872
1500
259
1196
2159
922
118
596
308
2184
1710
1588
897
127
2084
1261
1972
381
274
775
485
1476
282
432
1525
1714
778
289
320
1977
1616
1065
761
2127
2227
1570
1558
1563
2252
1833
628
429
1013
398
1321
510
1646
2234
99
1397
405
208
143
119
1390
829
231
519
2216
299
1790
89
577
1876
1539
1352
1619
1829
2249
284
1299
6
1877
2175
979
1518
727
732
2193
65
415
10
2181
1495
842
1781
2251
969
1631
1865
1930
1659
2098
354
1377
966
1219
355
120
1753
1806
1815
38
1268
1735
533
310
592
267
1516
1675
503
1704
2220
593
1647
1664
1090
937
2074
2189
1200
2200
773
1098
423
1604
283
160
433
1370
1499
49
296
1652
706
806
408
824
955
98
1490
794
359
62
2211
2228
681
756
134
1484
269
1717
42
1188
2241
449
1272
1021
1165
1934
2121
375
457
1358
1663
1015
351
1001
2207
2072
884
263
458
759
926
1349
2055
1346
1317
682
864
1151
30
1596
1099
402
1827
2215
728
751
583
471
1576
1173
1443
617
1069
388
1078
2187
1978
961
111
1058
2253
132
1583
2235
575
353
847
1281
2167
1483
2186
257
1283
2099
2107
965
2183
2110
81
915
1708
1448
436
1627
154
1111
559
1751
403
2062
1965
2005
340
614
1678
335
2109
150
2091
740
1225
212
1244
1559
802
1298
337
2255
2139
346
270
877
2044
211
\.



COPY "ExternalReviewer" ("ID", "assignedByReviewer") FROM stdin;
\.



COPY "Meta-Review" ("ID") FROM stdin;
\.



COPY "Meta-Reviewer" ("ID") FROM stdin;
\.



COPY "Paper" ("ID", "paperID", "rejectedBy", "readByMeta-Reviewer", "hasAuthor", title, "hasDecision", "acceptedBy") FROM stdin;
1558	296	\N	\N	1617	surges preheats loggias reconciliating photocatalyst	\N	\N
510	42	\N	\N	509	dwelling scrapped prat islanded burlily	\N	\N
1646	205	\N	\N	1106	teashop reaccepts gunneries exorcises pirog desexes	\N	\N
1585	140	\N	\N	2073	perspicuously disjoined nigglings midmonths labium peeped daydreams permuting	\N	\N
755	159	\N	\N	48	sodality outcrop slipcases phenylketonuria grunts venturers valiantly unremorsefully extradites	\N	\N
896	10	\N	\N	68	prat islanded burlily thanklessly swiveled polers oinked	\N	\N
345	87	\N	\N	344	debiting parrots paraguayans resonates overbites terminals giros podgy	\N	\N
1390	139	\N	\N	1333	mongoloids danker raunchiness perspicuously disjoined nigglings midmonths	\N	\N
1757	102	\N	\N	1715	bewitchment cheaters riffled kerneling napoleons rifer	\N	\N
1539	170	\N	\N	2006	tibetans slavishly hipless prs bluejays cuppier nonsurgical skimpiest	\N	\N
1366	237	\N	\N	1422	riffled kerneling napoleons rifer splinting surmisers satisfying undamped	\N	\N
1619	138	\N	\N	1618	retrogressions eliminates unknowns mongoloids danker raunchiness	\N	\N
727	41	\N	\N	52	richer succories dwelling scrapped	\N	\N
1385	209	\N	\N	1901	pirog desexes summable heliocentricity	\N	\N
833	203	\N	\N	831	evasiveness toto teashop reaccepts	\N	\N
42	3	\N	\N	2050	decentralizing hogfishes gantleted richer succories dwelling	\N	\N
864	137	\N	\N	1046	reducing nethermost nonfiction retrogressions eliminates	\N	\N
111	8	\N	\N	110	dwelling scrapped prat islanded burlily	\N	\N
558	179	\N	\N	1125	mongoloids danker raunchiness perspicuously disjoined	\N	\N
1010	196	\N	\N	1009	teratoid coadjutors thuggeries nondestructive	\N	\N
1184	265	\N	\N	1191	jambing sodality outcrop slipcases phenylketonuria grunts	\N	\N
374	172	\N	\N	372	outpoured dissociated heartier petitionee	\N	\N
771	66	\N	\N	790	kerneling napoleons rifer splinting surmisers	\N	\N
341	131	\N	\N	2124	brill neologic intermuscular fobbed transcribed	\N	\N
432	318	\N	\N	1337	cerebrum gloves mohairs admiralties bewigged playgoers	\N	\N
1935	27	\N	\N	\N	pirog desexes summable heliocentricity excretions recelebrating	\N	\N
2127	129	\N	\N	\N	nonsurgical skimpiest outpoured dissociated heartier petitionee brill neologic intermuscular	\N	\N
2227	97	\N	\N	\N	gloves mohairs admiralties bewigged playgoers cheques batting	\N	\N
1570	90	\N	\N	\N	apnea maxillary dumpers bering evasiveness toto	\N	\N
2199	50	\N	\N	\N	reaccepts gunneries exorcises pirog desexes summable heliocentricity	\N	\N
1563	273	\N	\N	\N	heliocentricity excretions recelebrating dually plateauing reoccupations	\N	\N
2252	270	\N	\N	\N	grunts venturers valiantly unremorsefully extradites	\N	\N
1833	310	\N	\N	\N	giros podgy vagus kinkiest xix recollected householder	\N	\N
1321	303	\N	\N	\N	valiantly unremorsefully extradites stollens ponderers conditione	\N	\N
1565	4	\N	\N	\N	hogfishes gantleted richer succories dwelling scrapped prat	\N	\N
810	83	\N	\N	\N	valiantly unremorsefully extradites stollens	\N	\N
1862	109	\N	\N	\N	unrealistic substituting surges preheats loggias reconciliating photocatalyst	\N	\N
2234	107	\N	\N	\N	funnyman commuted lachrymation floweret arcadian	\N	\N
99	136	\N	\N	\N	exec junketeers isolators reducing	\N	\N
2178	212	\N	\N	\N	heliocentricity excretions recelebrating dually plateauing reoccupations embossers	\N	\N
2088	170	\N	\N	\N	tibetans slavishly hipless prs bluejays cuppier nonsurgical skimpiest	\N	\N
2023	141	\N	\N	\N	midmonths labium peeped daydreams permuting immediately canzona interrelated cooked	\N	\N
1968	205	\N	\N	\N	teashop reaccepts gunneries exorcises pirog desexes	\N	\N
2238	275	\N	\N	\N	plateauing reoccupations embossers cerebrum gloves mohairs admiralties bewigged	\N	\N
2202	259	\N	\N	\N	preheats loggias reconciliating photocatalyst lenity tautological	\N	\N
1740	44	\N	\N	\N	burlily thanklessly swiveled polers oinked apnea maxillary	\N	\N
2240	92	\N	\N	\N	teashop reaccepts gunneries exorcises pirog desexes summable heliocentricity	\N	\N
2166	217	\N	\N	\N	reoccupations embossers cerebrum gloves mohairs admiralties	\N	\N
828	320	\N	\N	\N	cheques batting waspishly stilbestrol villainousness miscalling firefanged skeins	\N	\N
1397	74	\N	\N	\N	arcadian acridities unrealistic substituting surges preheats loggias	\N	\N
1843	99	\N	\N	\N	batting waspishly stilbestrol villainousness miscalling firefanged skeins equalled sandwiching	\N	\N
1783	248	\N	\N	\N	undermentioned outflanking funnyman commuted lachrymation floweret arcadian	\N	\N
2104	263	\N	\N	\N	lenity tautological jambing sodality	\N	\N
829	96	\N	\N	\N	reoccupations embossers cerebrum gloves mohairs admiralties	\N	\N
519	317	\N	\N	\N	plateauing reoccupations embossers cerebrum gloves	\N	\N
231	59	\N	\N	\N	playgoers cheques batting waspishly	\N	\N
2216	174	\N	\N	\N	transcribed swifters redigesting ostinato recalculation safest	\N	\N
1790	78	\N	\N	\N	photocatalyst lenity tautological jambing sodality	\N	\N
1907	357	\N	\N	\N	hops exec junketeers isolators reducing nethermost nonfiction retrogressions eliminates	\N	\N
89	123	\N	\N	\N	loftily yabber reprovingly blungers dwarflike effulgences coreless tuberculoses environs	\N	\N
577	178	\N	\N	\N	nonfiction retrogressions eliminates unknowns	\N	\N
2049	77	\N	\N	\N	loggias reconciliating photocatalyst lenity	\N	\N
1876	264	\N	\N	\N	tautological jambing sodality outcrop slipcases	\N	\N
2116	343	\N	\N	\N	objurgate treaties preprocessor despising loftily yabber reprovingly	\N	\N
1829	73	\N	\N	\N	lachrymation floweret arcadian acridities unrealistic substituting	\N	\N
950	91	\N	\N	\N	bering evasiveness toto teashop reaccepts gunneries exorcises	\N	\N
2249	115	\N	\N	\N	extradites stollens ponderers conditione loathly cancels debiting	\N	\N
439	54	\N	\N	\N	recelebrating dually plateauing reoccupations embossers	\N	\N
925	261	\N	\N	\N	reconciliating photocatalyst lenity tautological jambing sodality outcrop slipcases	\N	\N
1299	236	\N	\N	\N	cheaters riffled kerneling napoleons rifer splinting surmisers	\N	\N
6	296	\N	\N	\N	surges preheats loggias reconciliating photocatalyst	\N	\N
2254	283	\N	\N	\N	skeins equalled sandwiching bewitchment	\N	\N
1877	81	\N	\N	\N	slipcases phenylketonuria grunts venturers valiantly unremorsefully extradites stollens	\N	\N
1796	150	\N	\N	\N	miscalling firefanged skeins equalled sandwiching bewitchment	\N	\N
137	149	\N	\N	\N	batting waspishly stilbestrol villainousness miscalling	\N	\N
854	346	\N	\N	\N	effulgences coreless tuberculoses environs	\N	\N
983	354	\N	\N	\N	swifters redigesting ostinato recalculation safest signiory	\N	\N
2193	166	\N	\N	\N	objurgate treaties preprocessor despising	\N	\N
65	363	\N	\N	\N	peeped daydreams permuting immediately canzona interrelated cooked reformers goodwife	\N	\N
2051	314	\N	\N	\N	objurgate treaties preprocessor despising loftily	\N	\N
2155	111	\N	\N	\N	photocatalyst lenity tautological jambing sodality outcrop slipcases phenylketonuria grunts	\N	\N
2181	42	\N	\N	\N	dwelling scrapped prat islanded burlily	\N	\N
2218	187	\N	\N	\N	bedstraws thoughtless elation swampland earings circumscribed paralyzingly	\N	\N
1495	82	\N	\N	\N	grunts venturers valiantly unremorsefully extradites stollens ponderers conditione loathly	\N	\N
842	333	\N	\N	\N	tautological jambing sodality outcrop slipcases phenylketonuria grunts venturers valiantly	\N	\N
1781	235	\N	\N	\N	bewitchment cheaters riffled kerneling napoleons rifer	\N	\N
2251	1	\N	\N	\N	unmuzzling measles decentralizing hogfishes	\N	\N
176	291	\N	\N	\N	undermentioned outflanking funnyman commuted lachrymation floweret	\N	\N
1949	125	\N	\N	\N	coreless tuberculoses environs hulled preexamination	\N	\N
1865	197	\N	\N	\N	maladjustments subpartnership cordilleras recirculations alkalin	\N	\N
1930	14	\N	\N	\N	swiveled polers oinked apnea maxillary	\N	\N
2192	153	\N	\N	\N	surmisers satisfying undamped sharpers forbearer anesthetization undermentioned outflanking funnyman	\N	\N
1659	294	\N	\N	\N	arcadian acridities unrealistic substituting surges preheats loggias reconciliating photocatalyst	\N	\N
1946	365	\N	\N	\N	cooked reformers goodwife technicolor plenishes	\N	\N
2098	108	\N	\N	\N	floweret arcadian acridities unrealistic substituting surges	\N	\N
2168	47	\N	\N	\N	maxillary dumpers bering evasiveness	\N	\N
966	341	\N	\N	\N	vagus kinkiest xix recollected householder	\N	\N
2156	124	\N	\N	\N	blungers dwarflike effulgences coreless	\N	\N
1148	148	\N	\N	\N	admiralties bewigged playgoers cheques	\N	\N
1820	243	\N	\N	\N	satisfying undamped sharpers forbearer anesthetization undermentioned outflanking funnyman	\N	\N
1703	147	\N	\N	\N	embossers cerebrum gloves mohairs admiralties bewigged playgoers cheques batting	\N	\N
1059	207	\N	\N	\N	gunneries exorcises pirog desexes summable heliocentricity excretions recelebrating	\N	\N
1753	307	\N	\N	\N	debiting parrots paraguayans resonates	\N	\N
1797	328	\N	\N	\N	outflanking funnyman commuted lachrymation	\N	\N
1806	163	\N	\N	\N	paraguayans resonates overbites terminals giros podgy vagus	\N	\N
1815	189	\N	\N	\N	surrejoinders chestiest measurage tonsils pasturage thurifer teazle fringier rhythmical	\N	\N
1735	227	\N	\N	\N	waspishly stilbestrol villainousness miscalling	\N	\N
533	152	\N	\N	\N	kerneling napoleons rifer splinting surmisers satisfying undamped sharpers	\N	\N
1552	260	\N	\N	\N	loggias reconciliating photocatalyst lenity tautological jambing sodality	\N	\N
1516	345	\N	\N	\N	reprovingly blungers dwarflike effulgences coreless tuberculoses environs hulled preexamination	\N	\N
1675	49	\N	\N	\N	toto teashop reaccepts gunneries exorcises pirog	\N	\N
1690	134	\N	\N	\N	safest signiory latchets inflecting trephines hops exec junketeers	\N	\N
2220	241	\N	\N	\N	splinting surmisers satisfying undamped sharpers forbearer	\N	\N
593	232	\N	\N	\N	skeins equalled sandwiching bewitchment cheaters riffled kerneling napoleons rifer	\N	\N
1230	58	\N	\N	\N	admiralties bewigged playgoers cheques batting waspishly stilbestrol villainousness miscalling	\N	\N
1664	36	\N	\N	\N	embossers cerebrum gloves mohairs admiralties bewigged playgoers cheques batting	\N	\N
2134	2	\N	\N	\N	measles decentralizing hogfishes gantleted richer	\N	\N
2078	85	\N	\N	\N	ponderers conditione loathly cancels debiting parrots	\N	\N
937	5	\N	\N	\N	gantleted richer succories dwelling scrapped prat islanded burlily	\N	\N
2074	255	\N	\N	\N	acridities unrealistic substituting surges preheats loggias reconciliating photocatalyst	\N	\N
2189	140	\N	\N	\N	perspicuously disjoined nigglings midmonths labium peeped daydreams permuting	\N	\N
2200	246	\N	\N	\N	forbearer anesthetization undermentioned outflanking funnyman	\N	\N
1207	45	\N	\N	\N	swiveled polers oinked apnea maxillary dumpers bering evasiveness	\N	\N
423	237	\N	\N	\N	riffled kerneling napoleons rifer splinting surmisers satisfying undamped	\N	\N
1604	100	\N	\N	\N	villainousness miscalling firefanged skeins	\N	\N
1370	177	\N	\N	\N	junketeers isolators reducing nethermost nonfiction retrogressions eliminates unknowns mongoloids	\N	\N
1499	278	\N	\N	\N	admiralties bewigged playgoers cheques batting	\N	\N
296	309	\N	\N	\N	overbites terminals giros podgy vagus kinkiest	\N	\N
1652	281	\N	\N	\N	stilbestrol villainousness miscalling firefanged skeins equalled sandwiching bewitchment	\N	\N
1274	156	\N	\N	\N	arcadian acridities unrealistic substituting surges preheats	\N	\N
2048	301	\N	\N	\N	slipcases phenylketonuria grunts venturers	\N	\N
955	16	\N	\N	\N	oinked apnea maxillary dumpers bering evasiveness toto	\N	\N
794	70	\N	\N	\N	forbearer anesthetization undermentioned outflanking funnyman commuted lachrymation floweret arcadian	\N	\N
1490	265	\N	\N	\N	jambing sodality outcrop slipcases phenylketonuria grunts	\N	\N
2211	326	\N	\N	\N	satisfying undamped sharpers forbearer anesthetization undermentioned outflanking funnyman	\N	\N
2228	321	\N	\N	\N	stilbestrol villainousness miscalling firefanged skeins equalled sandwiching bewitchment cheaters	\N	\N
681	312	\N	\N	\N	xix recollected householder overeducated objurgate treaties preprocessor despising loftily	\N	\N
1822	60	\N	\N	\N	batting waspishly stilbestrol villainousness miscalling	\N	\N
1717	214	\N	\N	\N	recelebrating dually plateauing reoccupations embossers cerebrum gloves mohairs admiralties	\N	\N
2241	17	\N	\N	\N	apnea maxillary dumpers bering evasiveness toto teashop reaccepts	\N	\N
1832	269	\N	\N	\N	phenylketonuria grunts venturers valiantly	\N	\N
1105	362	\N	\N	\N	nigglings midmonths labium peeped daydreams permuting immediately canzona	\N	\N
1524	358	\N	\N	\N	isolators reducing nethermost nonfiction	\N	\N
1165	202	\N	\N	\N	zanier corkscrewed promiscuousness clewed	\N	\N
1934	298	\N	\N	\N	photocatalyst lenity tautological jambing sodality outcrop slipcases	\N	\N
2121	103	\N	\N	\N	kerneling napoleons rifer splinting surmisers satisfying undamped	\N	\N
1358	164	\N	\N	\N	giros podgy vagus kinkiest xix recollected householder overeducated	\N	\N
1663	219	\N	\N	\N	cerebrum gloves mohairs admiralties bewigged playgoers cheques batting	\N	\N
821	279	\N	\N	\N	playgoers cheques batting waspishly stilbestrol villainousness	\N	\N
1891	19	\N	\N	\N	dumpers bering evasiveness toto	\N	\N
1001	114	\N	\N	\N	venturers valiantly unremorsefully extradites stollens ponderers	\N	\N
2112	18	\N	\N	\N	maxillary dumpers bering evasiveness toto teashop reaccepts gunneries exorcises	\N	\N
2207	313	\N	\N	\N	householder overeducated objurgate treaties	\N	\N
2072	242	\N	\N	\N	surmisers satisfying undamped sharpers forbearer anesthetization undermentioned	\N	\N
884	194	\N	\N	\N	manifestly decaffeinates scalepan folklorists attractive yeller cognizably reminds	\N	\N
1126	122	\N	\N	\N	treaties preprocessor despising loftily yabber reprovingly blungers dwarflike	\N	\N
759	199	\N	\N	\N	neurosurgeon innervated hunts barrens emanative blowpipe varies	\N	\N
1458	172	\N	\N	\N	outpoured dissociated heartier petitionee	\N	\N
1941	352	\N	\N	\N	petitionee brill neologic intermuscular	\N	\N
1575	15	\N	\N	\N	polers oinked apnea maxillary dumpers bering	\N	\N
1349	329	\N	\N	\N	lachrymation floweret arcadian acridities unrealistic	\N	\N
2055	327	\N	\N	\N	forbearer anesthetization undermentioned outflanking funnyman commuted lachrymation floweret arcadian	\N	\N
1346	151	\N	\N	\N	sandwiching bewitchment cheaters riffled kerneling napoleons rifer	\N	\N
407	41	\N	\N	\N	richer succories dwelling scrapped	\N	\N
1610	366	\N	\N	\N	technicolor plenishes nippy bounden occulters blubberer	\N	\N
682	88	\N	\N	\N	islanded burlily thanklessly swiveled	\N	\N
1925	3	\N	\N	\N	decentralizing hogfishes gantleted richer succories dwelling	\N	\N
1871	251	\N	\N	\N	commuted lachrymation floweret arcadian	\N	\N
2077	268	\N	\N	\N	slipcases phenylketonuria grunts venturers valiantly unremorsefully extradites stollens ponderers	\N	\N
1622	126	\N	\N	\N	hulled preexamination oralogy tibetans slavishly hipless	\N	\N
2236	262	\N	\N	\N	photocatalyst lenity tautological jambing sodality outcrop slipcases phenylketonuria grunts	\N	\N
217	46	\N	\N	\N	oinked apnea maxillary dumpers bering evasiveness toto teashop reaccepts	\N	\N
2215	339	\N	\N	\N	paraguayans resonates overbites terminals giros podgy vagus kinkiest xix	\N	\N
1827	256	\N	\N	\N	unrealistic substituting surges preheats loggias reconciliating photocatalyst lenity tautological	\N	\N
728	160	\N	\N	\N	grunts venturers valiantly unremorsefully	\N	\N
751	186	\N	\N	\N	malines cliffhanger entailments reindexed bedstraws thoughtless	\N	\N
471	67	\N	\N	\N	rifer splinting surmisers satisfying undamped sharpers	\N	\N
177	20	\N	\N	\N	bering evasiveness toto teashop reaccepts	\N	\N
656	75	\N	\N	\N	unrealistic substituting surges preheats loggias reconciliating photocatalyst lenity	\N	\N
388	211	\N	\N	\N	summable heliocentricity excretions recelebrating dually plateauing	\N	\N
1078	37	\N	\N	\N	cerebrum gloves mohairs admiralties	\N	\N
2187	305	\N	\N	\N	ponderers conditione loathly cancels debiting parrots paraguayans resonates	\N	\N
1948	112	\N	\N	\N	jambing sodality outcrop slipcases	\N	\N
2140	13	\N	\N	\N	thanklessly swiveled polers oinked	\N	\N
1978	360	\N	\N	\N	unknowns mongoloids danker raunchiness perspicuously disjoined	\N	\N
1237	130	\N	\N	\N	dissociated heartier petitionee brill	\N	\N
961	29	\N	\N	\N	summable heliocentricity excretions recelebrating dually plateauing reoccupations embossers	\N	\N
1853	349	\N	\N	\N	hipless prs bluejays cuppier nonsurgical skimpiest outpoured	\N	\N
2253	12	\N	\N	\N	burlily thanklessly swiveled polers oinked apnea maxillary dumpers bering	\N	\N
1583	28	\N	\N	\N	desexes summable heliocentricity excretions recelebrating dually plateauing	\N	\N
797	266	\N	\N	\N	sodality outcrop slipcases phenylketonuria grunts venturers valiantly	\N	\N
2235	195	\N	\N	\N	attractive yeller cognizably reminds teratoid coadjutors thuggeries nondestructive maladjustments	\N	\N
1202	228	\N	\N	\N	stilbestrol villainousness miscalling firefanged skeins	\N	\N
1505	95	\N	\N	\N	recelebrating dually plateauing reoccupations embossers	\N	\N
620	62	\N	\N	\N	miscalling firefanged skeins equalled sandwiching bewitchment cheaters	\N	\N
2167	34	\N	\N	\N	plateauing reoccupations embossers cerebrum gloves mohairs admiralties	\N	\N
2186	68	\N	\N	\N	surmisers satisfying undamped sharpers forbearer anesthetization undermentioned	\N	\N
1283	169	\N	\N	\N	environs hulled preexamination oralogy tibetans slavishly hipless	\N	\N
498	304	\N	\N	\N	extradites stollens ponderers conditione loathly cancels debiting	\N	\N
2099	196	\N	\N	\N	teratoid coadjutors thuggeries nondestructive	\N	\N
2107	80	\N	\N	\N	sodality outcrop slipcases phenylketonuria grunts venturers valiantly	\N	\N
2020	306	\N	\N	\N	loathly cancels debiting parrots paraguayans resonates overbites terminals giros	\N	\N
999	367	\N	\N	\N	bounden occulters blubberer amenities desecrated tetrachlorides loutish	\N	\N
1653	10	\N	\N	\N	prat islanded burlily thanklessly swiveled polers oinked	\N	\N
1817	230	\N	\N	\N	miscalling firefanged skeins equalled sandwiching bewitchment cheaters	\N	\N
1888	94	\N	\N	\N	summable heliocentricity excretions recelebrating	\N	\N
2183	238	\N	\N	\N	kerneling napoleons rifer splinting surmisers satisfying undamped sharpers forbearer	\N	\N
2110	93	\N	\N	\N	exorcises pirog desexes summable heliocentricity excretions recelebrating dually plateauing	\N	\N
2036	221	\N	\N	\N	mohairs admiralties bewigged playgoers	\N	\N
1708	201	\N	\N	\N	machinability orbiters tormentor owner zanier corkscrewed promiscuousness clewed reassemble	\N	\N
1448	337	\N	\N	\N	ponderers conditione loathly cancels debiting parrots paraguayans	\N	\N
1304	53	\N	\N	\N	heliocentricity excretions recelebrating dually	\N	\N
2141	209	\N	\N	\N	pirog desexes summable heliocentricity	\N	\N
1627	139	\N	\N	\N	mongoloids danker raunchiness perspicuously disjoined nigglings midmonths	\N	\N
1909	104	\N	\N	\N	splinting surmisers satisfying undamped sharpers forbearer anesthetization undermentioned	\N	\N
403	224	\N	\N	\N	playgoers cheques batting waspishly stilbestrol villainousness miscalling	\N	\N
2062	175	\N	\N	\N	recalculation safest signiory latchets inflecting trephines hops	\N	\N
2108	276	\N	\N	\N	embossers cerebrum gloves mohairs admiralties bewigged playgoers cheques batting	\N	\N
1965	280	\N	\N	\N	batting waspishly stilbestrol villainousness miscalling firefanged skeins	\N	\N
2005	226	\N	\N	\N	batting waspishly stilbestrol villainousness miscalling firefanged skeins equalled sandwiching	\N	\N
988	101	\N	\N	\N	skeins equalled sandwiching bewitchment cheaters	\N	\N
614	245	\N	\N	\N	sharpers forbearer anesthetization undermentioned	\N	\N
1678	361	\N	\N	\N	raunchiness perspicuously disjoined nigglings midmonths labium peeped	\N	\N
2123	116	\N	\N	\N	conditione loathly cancels debiting parrots paraguayans resonates overbites	\N	\N
2109	7	\N	\N	\N	succories dwelling scrapped prat	\N	\N
1179	215	\N	\N	\N	dually plateauing reoccupations embossers	\N	\N
2091	348	\N	\N	\N	oralogy tibetans slavishly hipless prs bluejays	\N	\N
25	38	\N	\N	\N	gloves mohairs admiralties bewigged playgoers	\N	\N
740	173	\N	\N	\N	brill neologic intermuscular fobbed transcribed	\N	\N
1388	244	\N	\N	\N	undamped sharpers forbearer anesthetization undermentioned outflanking funnyman commuted lachrymation	\N	\N
1172	290	\N	\N	\N	forbearer anesthetization undermentioned outflanking funnyman	\N	\N
2162	204	\N	\N	\N	toto teashop reaccepts gunneries exorcises	\N	\N
1656	206	\N	\N	\N	reaccepts gunneries exorcises pirog desexes summable heliocentricity	\N	\N
2086	335	\N	\N	\N	grunts venturers valiantly unremorsefully extradites	\N	\N
337	157	\N	\N	\N	surges preheats loggias reconciliating photocatalyst lenity tautological	\N	\N
1857	184	\N	\N	\N	bounden occulters blubberer amenities	\N	\N
2255	208	\N	\N	\N	exorcises pirog desexes summable heliocentricity excretions recelebrating dually plateauing	\N	\N
2096	183	\N	\N	\N	goodwife technicolor plenishes nippy bounden occulters blubberer amenities desecrated	\N	\N
2139	284	\N	\N	\N	sandwiching bewitchment cheaters riffled kerneling	\N	\N
1245	308	\N	\N	\N	paraguayans resonates overbites terminals giros	\N	\N
1772	168	\N	\N	\N	dwarflike effulgences coreless tuberculoses environs hulled	\N	\N
1766	155	\N	\N	\N	funnyman commuted lachrymation floweret arcadian	\N	\N
2208	89	\N	\N	\N	swiveled polers oinked apnea maxillary	\N	\N
877	106	\N	\N	\N	anesthetization undermentioned outflanking funnyman	\N	\N
2044	64	\N	\N	\N	sandwiching bewitchment cheaters riffled kerneling napoleons rifer splinting surmisers	\N	\N
1980	120	\N	\N	\N	kinkiest xix recollected householder overeducated objurgate	\N	\N
2147	39	\N	\N	\N	mohairs admiralties bewigged playgoers cheques batting	\N	\N
2231	347	\N	\N	\N	environs hulled preexamination oralogy tibetans	\N	\N
1880	277	\N	\N	\N	gloves mohairs admiralties bewigged	\N	\N
1747	57	\N	\N	\N	gloves mohairs admiralties bewigged playgoers cheques batting waspishly	\N	\N
2164	21	\N	\N	\N	evasiveness toto teashop reaccepts gunneries exorcises	\N	\N
2033	167	\N	\N	\N	loftily yabber reprovingly blungers dwarflike	\N	\N
2221	254	\N	\N	\N	arcadian acridities unrealistic substituting surges preheats loggias	\N	\N
1658	23	\N	\N	\N	teashop reaccepts gunneries exorcises pirog desexes summable heliocentricity	\N	\N
2244	355	\N	\N	\N	recalculation safest signiory latchets inflecting trephines hops	\N	\N
1396	257	\N	\N	\N	substituting surges preheats loggias	\N	\N
1527	193	\N	\N	\N	irades inadvisability disciplinarians majors manifestly decaffeinates scalepan	\N	\N
2222	71	\N	\N	\N	undermentioned outflanking funnyman commuted	\N	\N
2002	213	\N	\N	\N	excretions recelebrating dually plateauing reoccupations embossers cerebrum gloves	\N	\N
1293	40	\N	\N	\N	admiralties bewigged playgoers cheques batting waspishly stilbestrol	\N	\N
2040	51	\N	\N	\N	exorcises pirog desexes summable heliocentricity excretions recelebrating dually	\N	\N
2133	271	\N	\N	\N	exorcises pirog desexes summable	\N	\N
2250	128	\N	\N	\N	prs bluejays cuppier nonsurgical skimpiest outpoured dissociated heartier	\N	\N
1848	336	\N	\N	\N	unremorsefully extradites stollens ponderers conditione loathly	\N	\N
2225	84	\N	\N	\N	extradites stollens ponderers conditione loathly	\N	\N
2174	145	\N	\N	\N	desexes summable heliocentricity excretions recelebrating dually plateauing	\N	\N
2242	288	\N	\N	\N	surmisers satisfying undamped sharpers forbearer anesthetization undermentioned outflanking funnyman	\N	\N
1893	331	\N	\N	\N	surges preheats loggias reconciliating photocatalyst lenity tautological	\N	\N
2233	182	\N	\N	\N	canzona interrelated cooked reformers goodwife technicolor plenishes nippy	\N	\N
2153	364	\N	\N	\N	immediately canzona interrelated cooked	\N	\N
43	304	\N	\N	48	extradites stollens ponderers conditione loathly cancels debiting	\N	\N
66	317	\N	\N	68	plateauing reoccupations embossers cerebrum gloves	\N	\N
765	282	\N	\N	768	miscalling firefanged skeins equalled sandwiching bewitchment cheaters riffled kerneling	\N	\N
923	54	\N	\N	768	recelebrating dually plateauing reoccupations embossers	\N	\N
924	28	\N	\N	768	desexes summable heliocentricity excretions recelebrating dually plateauing	\N	\N
883	148	\N	\N	768	admiralties bewigged playgoers cheques	\N	\N
259	295	\N	\N	796	unrealistic substituting surges preheats	\N	\N
49	2	\N	\N	52	measles decentralizing hogfishes gantleted richer	\N	\N
284	338	\N	\N	286	cancels debiting parrots paraguayans resonates overbites terminals giros	\N	\N
1188	257	\N	\N	1189	substituting surges preheats loggias	\N	\N
606	274	\N	\N	609	recelebrating dually plateauing reoccupations embossers cerebrum gloves	\N	\N
888	144	\N	\N	609	reaccepts gunneries exorcises pirog desexes summable	\N	\N
140	91	\N	\N	609	bering evasiveness toto teashop reaccepts gunneries exorcises	\N	\N
143	210	\N	\N	609	desexes summable heliocentricity excretions recelebrating	\N	\N
375	240	\N	\N	377	rifer splinting surmisers satisfying undamped	\N	\N
98	119	\N	\N	94	giros podgy vagus kinkiest xix	\N	\N
754	130	\N	\N	982	dissociated heartier petitionee brill	\N	\N
453	100	\N	\N	1422	villainousness miscalling firefanged skeins	\N	\N
906	202	\N	\N	907	zanier corkscrewed promiscuousness clewed	\N	\N
926	215	\N	\N	928	dually plateauing reoccupations embossers	\N	\N
37	178	\N	\N	108	nonfiction retrogressions eliminates unknowns	\N	\N
396	277	\N	\N	32	gloves mohairs admiralties bewigged	\N	\N
1359	118	\N	\N	1361	resonates overbites terminals giros	\N	\N
2102	217	\N	\N	461	reoccupations embossers cerebrum gloves mohairs admiralties	\N	\N
915	271	\N	\N	992	exorcises pirog desexes summable	\N	\N
1082	48	\N	\N	1083	bering evasiveness toto teashop reaccepts	\N	\N
405	184	\N	\N	1761	bounden occulters blubberer amenities	\N	\N
1730	53	\N	\N	1477	heliocentricity excretions recelebrating dually	\N	\N
1029	328	\N	\N	1680	outflanking funnyman commuted lachrymation	\N	\N
295	307	\N	\N	1446	debiting parrots paraguayans resonates	\N	\N
443	65	\N	\N	446	cheaters riffled kerneling napoleons	\N	\N
969	347	\N	\N	2158	environs hulled preexamination oralogy tibetans	\N	\N
669	341	\N	\N	672	vagus kinkiest xix recollected householder	\N	\N
2039	234	\N	\N	\N	sandwiching bewitchment cheaters riffled kerneling	\N	\N
2237	6	\N	\N	\N	richer succories dwelling scrapped prat islanded burlily thanklessly swiveled	\N	\N
2143	297	\N	\N	\N	loggias reconciliating photocatalyst lenity tautological jambing	\N	\N
1782	52	\N	\N	\N	desexes summable heliocentricity excretions recelebrating dually plateauing reoccupations embossers	\N	\N
1995	159	\N	\N	\N	sodality outcrop slipcases phenylketonuria grunts venturers valiantly unremorsefully extradites	\N	\N
2111	289	\N	\N	\N	undamped sharpers forbearer anesthetization	\N	\N
2195	220	\N	\N	\N	gloves mohairs admiralties bewigged playgoers cheques batting waspishly stilbestrol	\N	\N
737	338	\N	\N	\N	cancels debiting parrots paraguayans resonates overbites terminals giros	\N	\N
1391	61	\N	\N	\N	stilbestrol villainousness miscalling firefanged skeins equalled	\N	\N
1975	35	\N	\N	\N	reoccupations embossers cerebrum gloves mohairs admiralties bewigged playgoers	\N	\N
1136	135	\N	\N	\N	inflecting trephines hops exec junketeers isolators reducing nethermost nonfiction	\N	\N
1463	185	\N	\N	\N	desecrated tetrachlorides loutish polygony malines	\N	\N
2214	161	\N	\N	\N	extradites stollens ponderers conditione loathly	\N	\N
2203	300	\N	\N	\N	sodality outcrop slipcases phenylketonuria grunts venturers valiantly unremorsefully extradites	\N	\N
1879	8	\N	\N	\N	dwelling scrapped prat islanded burlily	\N	\N
2003	30	\N	\N	\N	heliocentricity excretions recelebrating dually plateauing reoccupations embossers cerebrum gloves	\N	\N
193	33	\N	\N	\N	dually plateauing reoccupations embossers cerebrum gloves	\N	\N
1521	180	\N	\N	\N	disjoined nigglings midmonths labium peeped daydreams	\N	\N
2201	63	\N	\N	\N	skeins equalled sandwiching bewitchment cheaters riffled kerneling napoleons	\N	\N
2163	292	\N	\N	\N	funnyman commuted lachrymation floweret arcadian acridities unrealistic	\N	\N
1322	79	\N	\N	\N	tautological jambing sodality outcrop slipcases phenylketonuria	\N	\N
1314	66	\N	\N	\N	kerneling napoleons rifer splinting surmisers	\N	\N
1728	31	\N	\N	\N	excretions recelebrating dually plateauing	\N	\N
190	216	\N	\N	\N	plateauing reoccupations embossers cerebrum gloves	\N	\N
2248	190	\N	\N	\N	pasturage thurifer teazle fringier	\N	\N
1911	192	\N	\N	\N	inserter plights corollaries gaudiness irades inadvisability	\N	\N
2179	225	\N	\N	\N	cheques batting waspishly stilbestrol villainousness miscalling firefanged skeins	\N	\N
1590	340	\N	\N	\N	terminals giros podgy vagus	\N	\N
1914	117	\N	\N	\N	debiting parrots paraguayans resonates overbites terminals giros podgy vagus	\N	\N
1835	86	\N	\N	\N	loathly cancels debiting parrots paraguayans resonates overbites	\N	\N
2015	56	\N	\N	\N	embossers cerebrum gloves mohairs admiralties bewigged playgoers	\N	\N
2011	43	\N	\N	\N	prat islanded burlily thanklessly swiveled polers	\N	\N
552	165	\N	\N	\N	xix recollected householder overeducated objurgate treaties preprocessor despising loftily	\N	\N
936	274	\N	\N	\N	recelebrating dually plateauing reoccupations embossers cerebrum gloves	\N	\N
2076	72	\N	\N	\N	funnyman commuted lachrymation floweret arcadian	\N	\N
1684	121	\N	\N	\N	householder overeducated objurgate treaties preprocessor despising loftily	\N	\N
1048	11	\N	\N	\N	islanded burlily thanklessly swiveled polers oinked apnea maxillary	\N	\N
2120	356	\N	\N	\N	latchets inflecting trephines hops exec junketeers isolators reducing	\N	\N
1535	118	\N	\N	\N	resonates overbites terminals giros	\N	\N
2191	318	\N	\N	\N	cerebrum gloves mohairs admiralties bewigged playgoers	\N	\N
313	144	\N	\N	\N	reaccepts gunneries exorcises pirog desexes summable	\N	\N
2024	319	\N	\N	\N	admiralties bewigged playgoers cheques batting waspishly stilbestrol	\N	\N
2243	342	\N	\N	\N	recollected householder overeducated objurgate treaties preprocessor	\N	\N
2224	267	\N	\N	\N	outcrop slipcases phenylketonuria grunts venturers valiantly unremorsefully extradites	\N	\N
1067	295	\N	\N	\N	unrealistic substituting surges preheats	\N	\N
1929	285	\N	\N	\N	cheaters riffled kerneling napoleons rifer splinting	\N	\N
1897	344	\N	\N	\N	despising loftily yabber reprovingly blungers dwarflike effulgences coreless	\N	\N
2041	247	\N	\N	\N	anesthetization undermentioned outflanking funnyman commuted lachrymation	\N	\N
1956	138	\N	\N	\N	retrogressions eliminates unknowns mongoloids danker raunchiness	\N	\N
1982	133	\N	\N	\N	redigesting ostinato recalculation safest signiory latchets inflecting	\N	\N
1510	332	\N	\N	\N	reconciliating photocatalyst lenity tautological jambing sodality outcrop slipcases	\N	\N
1429	324	\N	\N	\N	riffled kerneling napoleons rifer splinting surmisers	\N	\N
1850	171	\N	\N	\N	bluejays cuppier nonsurgical skimpiest outpoured dissociated heartier petitionee brill	\N	\N
1999	200	\N	\N	\N	emanative blowpipe varies thickest machinability orbiters tormentor owner	\N	\N
1785	65	\N	\N	\N	cheaters riffled kerneling napoleons	\N	\N
2082	322	\N	\N	\N	firefanged skeins equalled sandwiching	\N	\N
1789	302	\N	\N	\N	grunts venturers valiantly unremorsefully extradites	\N	\N
2122	191	\N	\N	\N	rhythmical wastebaskets powderer immigrates inserter	\N	\N
1793	315	\N	\N	\N	preprocessor despising loftily yabber reprovingly blungers	\N	\N
2114	330	\N	\N	\N	acridities unrealistic substituting surges preheats loggias	\N	\N
2013	239	\N	\N	\N	napoleons rifer splinting surmisers	\N	\N
2230	334	\N	\N	\N	outcrop slipcases phenylketonuria grunts	\N	\N
1381	231	\N	\N	\N	firefanged skeins equalled sandwiching bewitchment cheaters riffled kerneling	\N	\N
2172	132	\N	\N	\N	fobbed transcribed swifters redigesting ostinato recalculation	\N	\N
1584	299	\N	\N	\N	tautological jambing sodality outcrop slipcases phenylketonuria grunts venturers	\N	\N
862	181	\N	\N	\N	peeped daydreams permuting immediately canzona interrelated cooked	\N	\N
2144	26	\N	\N	\N	exorcises pirog desexes summable heliocentricity	\N	\N
1611	113	\N	\N	\N	slipcases phenylketonuria grunts venturers valiantly	\N	\N
1292	223	\N	\N	\N	bewigged playgoers cheques batting waspishly stilbestrol	\N	\N
1574	286	\N	\N	\N	kerneling napoleons rifer splinting surmisers satisfying undamped	\N	\N
2067	188	\N	\N	\N	earings circumscribed paralyzingly pouchy surrejoinders chestiest measurage tonsils	\N	\N
1332	316	\N	\N	\N	excretions recelebrating dually plateauing	\N	\N
2103	250	\N	\N	\N	funnyman commuted lachrymation floweret arcadian acridities unrealistic substituting surges	\N	\N
39	350	\N	\N	\N	cuppier nonsurgical skimpiest outpoured dissociated heartier petitionee brill	\N	\N
1285	210	\N	\N	\N	desexes summable heliocentricity excretions recelebrating	\N	\N
786	253	\N	\N	\N	floweret arcadian acridities unrealistic substituting surges	\N	\N
1974	218	\N	\N	\N	embossers cerebrum gloves mohairs admiralties bewigged playgoers	\N	\N
1988	146	\N	\N	\N	recelebrating dually plateauing reoccupations embossers cerebrum gloves mohairs	\N	\N
1193	32	\N	\N	\N	recelebrating dually plateauing reoccupations embossers	\N	\N
1480	252	\N	\N	\N	lachrymation floweret arcadian acridities unrealistic	\N	\N
2027	154	\N	\N	\N	forbearer anesthetization undermentioned outflanking	\N	\N
1927	272	\N	\N	\N	desexes summable heliocentricity excretions recelebrating	\N	\N
712	325	\N	\N	\N	rifer splinting surmisers satisfying undamped sharpers forbearer	\N	\N
1312	249	\N	\N	\N	outflanking funnyman commuted lachrymation floweret arcadian acridities unrealistic	\N	\N
2205	76	\N	\N	\N	surges preheats loggias reconciliating photocatalyst lenity tautological jambing sodality	\N	\N
954	287	\N	\N	\N	rifer splinting surmisers satisfying undamped sharpers forbearer anesthetization	\N	\N
1411	311	\N	\N	\N	vagus kinkiest xix recollected householder overeducated objurgate treaties	\N	\N
2148	351	\N	\N	\N	outpoured dissociated heartier petitionee brill neologic intermuscular fobbed transcribed	\N	\N
1811	176	\N	\N	\N	inflecting trephines hops exec junketeers isolators reducing nethermost	\N	\N
72	110	\N	\N	\N	preheats loggias reconciliating photocatalyst lenity tautological jambing sodality	\N	\N
1602	353	\N	\N	\N	intermuscular fobbed transcribed swifters redigesting	\N	\N
2058	162	\N	\N	\N	loathly cancels debiting parrots paraguayans resonates	\N	\N
1413	87	\N	\N	\N	debiting parrots paraguayans resonates overbites terminals giros podgy	\N	\N
646	258	\N	\N	\N	surges preheats loggias reconciliating photocatalyst	\N	\N
1777	222	\N	\N	\N	admiralties bewigged playgoers cheques batting	\N	\N
872	359	\N	\N	\N	nonfiction retrogressions eliminates unknowns mongoloids	\N	\N
1500	9	\N	\N	\N	scrapped prat islanded burlily thanklessly swiveled	\N	\N
2159	137	\N	\N	\N	reducing nethermost nonfiction retrogressions eliminates	\N	\N
2018	69	\N	\N	\N	undamped sharpers forbearer anesthetization undermentioned outflanking funnyman commuted	\N	\N
2184	55	\N	\N	\N	plateauing reoccupations embossers cerebrum gloves mohairs	\N	\N
1758	142	\N	\N	\N	oinked apnea maxillary dumpers	\N	\N
1047	282	\N	\N	\N	miscalling firefanged skeins equalled sandwiching bewitchment cheaters riffled kerneling	\N	\N
897	119	\N	\N	\N	giros podgy vagus kinkiest xix	\N	\N
7	25	\N	\N	\N	gunneries exorcises pirog desexes	\N	\N
171	203	\N	\N	\N	evasiveness toto teashop reaccepts	\N	\N
2084	98	\N	\N	\N	bewigged playgoers cheques batting waspishly stilbestrol villainousness miscalling	\N	\N
1261	143	\N	\N	\N	bering evasiveness toto teashop reaccepts	\N	\N
2246	102	\N	\N	\N	bewitchment cheaters riffled kerneling napoleons rifer	\N	\N
2197	233	\N	\N	\N	equalled sandwiching bewitchment cheaters	\N	\N
1972	22	\N	\N	\N	toto teashop reaccepts gunneries exorcises pirog desexes	\N	\N
1971	293	\N	\N	\N	lachrymation floweret arcadian acridities unrealistic substituting surges preheats	\N	\N
1474	198	\N	\N	\N	alkalin succulently marquise underlaid neurosurgeon innervated	\N	\N
1014	127	\N	\N	\N	tibetans slavishly hipless prs bluejays cuppier nonsurgical	\N	\N
1841	323	\N	\N	\N	sandwiching bewitchment cheaters riffled kerneling	\N	\N
1525	105	\N	\N	\N	undamped sharpers forbearer anesthetization undermentioned outflanking funnyman commuted lachrymation	\N	\N
289	131	\N	\N	\N	brill neologic intermuscular fobbed transcribed	\N	\N
1977	48	\N	\N	\N	bering evasiveness toto teashop reaccepts	\N	\N
1243	61	\N	\N	377	stilbestrol villainousness miscalling firefanged skeins equalled	\N	\N
1518	239	\N	\N	1517	napoleons rifer splinting surmisers	\N	\N
1159	15	\N	\N	1878	polers oinked apnea maxillary dumpers bering	\N	\N
1057	252	\N	\N	796	lachrymation floweret arcadian acridities unrealistic	\N	\N
714	168	\N	\N	713	dwarflike effulgences coreless tuberculoses environs hulled	\N	\N
129	50	\N	\N	1305	reaccepts gunneries exorcises pirog desexes summable heliocentricity	\N	\N
1771	313	\N	\N	1399	householder overeducated objurgate treaties	\N	\N
1631	360	\N	\N	1654	unknowns mongoloids danker raunchiness perspicuously disjoined	\N	\N
333	322	\N	\N	332	firefanged skeins equalled sandwiching	\N	\N
354	110	\N	\N	2182	preheats loggias reconciliating photocatalyst lenity tautological jambing sodality	\N	\N
1450	356	\N	\N	1449	latchets inflecting trephines hops exec junketeers isolators reducing	\N	\N
349	96	\N	\N	348	reoccupations embossers cerebrum gloves mohairs admiralties	\N	\N
1192	336	\N	\N	1191	unremorsefully extradites stollens ponderers conditione loathly	\N	\N
59	71	\N	\N	1378	undermentioned outflanking funnyman commuted	\N	\N
1197	262	\N	\N	1392	photocatalyst lenity tautological jambing sodality outcrop slipcases phenylketonuria grunts	\N	\N
1030	242	\N	\N	86	surmisers satisfying undamped sharpers forbearer anesthetization undermentioned	\N	\N
38	136	\N	\N	2061	exec junketeers isolators reducing	\N	\N
1268	344	\N	\N	1553	despising loftily yabber reprovingly blungers dwarflike effulgences coreless	\N	\N
592	342	\N	\N	591	recollected householder overeducated objurgate treaties preprocessor	\N	\N
1647	7	\N	\N	1106	succories dwelling scrapped prat	\N	\N
1032	62	\N	\N	86	miscalling firefanged skeins equalled sandwiching bewitchment cheaters	\N	\N
1200	84	\N	\N	224	extradites stollens ponderers conditione loathly	\N	\N
773	250	\N	\N	790	funnyman commuted lachrymation floweret arcadian acridities unrealistic substituting surges	\N	\N
283	181	\N	\N	281	peeped daydreams permuting immediately canzona interrelated cooked	\N	\N
435	275	\N	\N	943	plateauing reoccupations embossers cerebrum gloves mohairs admiralties bewigged	\N	\N
433	212	\N	\N	943	heliocentricity excretions recelebrating dually plateauing reoccupations embossers	\N	\N
895	208	\N	\N	68	exorcises pirog desexes summable heliocentricity excretions recelebrating dually plateauing	\N	\N
581	259	\N	\N	1236	preheats loggias reconciliating photocatalyst lenity tautological	\N	\N
806	126	\N	\N	805	hulled preexamination oralogy tibetans slavishly hipless	\N	\N
309	305	\N	\N	438	ponderers conditione loathly cancels debiting parrots paraguayans resonates	\N	\N
756	111	\N	\N	48	photocatalyst lenity tautological jambing sodality outcrop slipcases phenylketonuria grunts	\N	\N
705	129	\N	\N	1473	nonsurgical skimpiest outpoured dissociated heartier petitionee brill neologic intermuscular	\N	\N
1557	68	\N	\N	1617	surmisers satisfying undamped sharpers forbearer anesthetization undermentioned	\N	\N
260	67	\N	\N	796	rifer splinting surmisers satisfying undamped sharpers	\N	\N
1363	59	\N	\N	627	playgoers cheques batting waspishly	\N	\N
512	142	\N	\N	509	oinked apnea maxillary dumpers	\N	\N
457	288	\N	\N	610	surmisers satisfying undamped sharpers forbearer anesthetization undermentioned outflanking funnyman	\N	\N
776	154	\N	\N	790	forbearer anesthetization undermentioned outflanking	\N	\N
351	225	\N	\N	348	cheques batting waspishly stilbestrol villainousness miscalling firefanged skeins	\N	\N
857	90	\N	\N	789	apnea maxillary dumpers bering evasiveness toto	\N	\N
124	270	\N	\N	48	grunts venturers valiantly unremorsefully extradites	\N	\N
1317	146	\N	\N	1305	recelebrating dually plateauing reoccupations embossers cerebrum gloves mohairs	\N	\N
1151	141	\N	\N	1562	midmonths labium peeped daydreams permuting immediately canzona interrelated cooked	\N	\N
1596	38	\N	\N	627	gloves mohairs admiralties bewigged playgoers	\N	\N
689	190	\N	\N	2212	pasturage thurifer teazle fringier	\N	\N
947	13	\N	\N	1337	thanklessly swiveled polers oinked	\N	\N
832	5	\N	\N	831	gantleted richer succories dwelling scrapped prat islanded burlily	\N	\N
1324	149	\N	\N	1211	batting waspishly stilbestrol villainousness miscalling	\N	\N
356	74	\N	\N	412	arcadian acridities unrealistic substituting surges preheats loggias	\N	\N
347	315	\N	\N	344	preprocessor despising loftily yabber reprovingly blungers	\N	\N
350	27	\N	\N	348	pirog desexes summable heliocentricity excretions recelebrating	\N	\N
1058	105	\N	\N	796	undamped sharpers forbearer anesthetization undermentioned outflanking funnyman commuted lachrymation	\N	\N
1384	11	\N	\N	1901	islanded burlily thanklessly swiveled polers oinked apnea maxillary	\N	\N
19	188	\N	\N	1607	earings circumscribed paralyzingly pouchy surrejoinders chestiest measurage tonsils	\N	\N
1587	220	\N	\N	441	gloves mohairs admiralties bewigged playgoers cheques batting waspishly stilbestrol	\N	\N
132	218	\N	\N	1305	embossers cerebrum gloves mohairs admiralties bewigged playgoers	\N	\N
113	272	\N	\N	110	desexes summable heliocentricity excretions recelebrating	\N	\N
575	174	\N	\N	2124	transcribed swifters redigesting ostinato recalculation safest	\N	\N
353	158	\N	\N	412	photocatalyst lenity tautological jambing sodality outcrop slipcases phenylketonuria	\N	\N
264	326	\N	\N	1651	satisfying undamped sharpers forbearer anesthetization undermentioned outflanking funnyman	\N	\N
257	332	\N	\N	392	reconciliating photocatalyst lenity tautological jambing sodality outcrop slipcases	\N	\N
965	195	\N	\N	964	attractive yeller cognizably reminds teratoid coadjutors thuggeries nondestructive maladjustments	\N	\N
436	47	\N	\N	943	maxillary dumpers bering evasiveness	\N	\N
133	20	\N	\N	1305	bering evasiveness toto teashop reaccepts	\N	\N
1111	348	\N	\N	1994	oralogy tibetans slavishly hipless prs bluejays	\N	\N
1751	285	\N	\N	165	cheaters riffled kerneling napoleons rifer splinting	\N	\N
559	297	\N	\N	489	loggias reconciliating photocatalyst lenity tautological jambing	\N	\N
334	25	\N	\N	332	gunneries exorcises pirog desexes	\N	\N
380	330	\N	\N	379	acridities unrealistic substituting surges preheats loggias	\N	\N
340	116	\N	\N	1180	conditione loathly cancels debiting parrots paraguayans resonates overbites	\N	\N
785	187	\N	\N	784	bedstraws thoughtless elation swampland earings circumscribed paralyzingly	\N	\N
1532	246	\N	\N	1308	forbearer anesthetization undermentioned outflanking funnyman	\N	\N
335	223	\N	\N	332	bewigged playgoers cheques batting waspishly stilbestrol	\N	\N
150	314	\N	\N	662	objurgate treaties preprocessor despising loftily	\N	\N
781	334	\N	\N	1236	outcrop slipcases phenylketonuria grunts	\N	\N
1225	97	\N	\N	940	gloves mohairs admiralties bewigged playgoers cheques batting	\N	\N
212	104	\N	\N	1070	splinting surmisers satisfying undamped sharpers forbearer anesthetization undermentioned	\N	\N
1244	289	\N	\N	377	undamped sharpers forbearer anesthetization	\N	\N
1559	254	\N	\N	1617	arcadian acridities unrealistic substituting surges preheats loggias	\N	\N
949	211	\N	\N	1337	summable heliocentricity excretions recelebrating dually plateauing	\N	\N
346	345	\N	\N	344	reprovingly blungers dwarflike effulgences coreless tuberculoses environs hulled preexamination	\N	\N
659	351	\N	\N	1016	outpoured dissociated heartier petitionee brill neologic intermuscular fobbed transcribed	\N	\N
1140	83	\N	\N	1221	valiantly unremorsefully extradites stollens	\N	\N
623	115	\N	\N	1745	extradites stollens ponderers conditione loathly cancels debiting	\N	\N
571	55	\N	\N	940	plateauing reoccupations embossers cerebrum gloves mohairs	\N	\N
827	120	\N	\N	1895	kinkiest xix recollected householder overeducated objurgate	\N	\N
979	310	\N	\N	1745	giros podgy vagus kinkiest xix recollected householder	\N	\N
976	162	\N	\N	1745	loathly cancels debiting parrots paraguayans resonates	\N	\N
474	78	\N	\N	278	photocatalyst lenity tautological jambing sodality	\N	\N
1272	306	\N	\N	278	loathly cancels debiting parrots paraguayans resonates overbites terminals giros	\N	\N
274	160	\N	\N	278	grunts venturers valiantly unremorsefully	\N	\N
192	361	\N	\N	1507	raunchiness perspicuously disjoined nigglings midmonths labium peeped	\N	\N
1620	88	\N	\N	2050	islanded burlily thanklessly swiveled	\N	\N
1099	1	\N	\N	1100	unmuzzling measles decentralizing hogfishes	\N	\N
628	73	\N	\N	630	lachrymation floweret arcadian acridities unrealistic substituting	\N	\N
1182	301	\N	\N	630	slipcases phenylketonuria grunts venturers	\N	\N
1117	264	\N	\N	630	tautological jambing sodality outcrop slipcases	\N	\N
1021	109	\N	\N	630	unrealistic substituting surges preheats loggias reconciliating photocatalyst	\N	\N
1298	51	\N	\N	441	exorcises pirog desexes summable heliocentricity excretions recelebrating dually	\N	\N
440	279	\N	\N	441	playgoers cheques batting waspishly stilbestrol villainousness	\N	\N
1051	22	\N	\N	441	toto teashop reaccepts gunneries exorcises pirog desexes	\N	\N
62	261	\N	\N	588	reconciliating photocatalyst lenity tautological jambing sodality outcrop slipcases	\N	\N
1038	108	\N	\N	588	floweret arcadian acridities unrealistic substituting surges	\N	\N
1714	243	\N	\N	1715	satisfying undamped sharpers forbearer anesthetization undermentioned outflanking funnyman	\N	\N
458	320	\N	\N	461	cheques batting waspishly stilbestrol villainousness miscalling firefanged skeins	\N	\N
824	19	\N	\N	461	dumpers bering evasiveness toto	\N	\N
1281	353	\N	\N	1418	intermuscular fobbed transcribed swifters redigesting	\N	\N
1484	89	\N	\N	992	swiveled polers oinked apnea maxillary	\N	\N
991	6	\N	\N	992	richer succories dwelling scrapped prat islanded burlily thanklessly swiveled	\N	\N
914	43	\N	\N	992	prat islanded burlily thanklessly swiveled polers	\N	\N
1483	204	\N	\N	992	toto teashop reaccepts gunneries exorcises	\N	\N
514	194	\N	\N	551	manifestly decaffeinates scalepan folklorists attractive yeller cognizably reminds	\N	\N
360	199	\N	\N	362	neurosurgeon innervated hunts barrens emanative blowpipe varies	\N	\N
694	354	\N	\N	695	swifters redigesting ostinato recalculation safest signiory	\N	\N
913	191	\N	\N	2056	rhythmical wastebaskets powderer immigrates inserter	\N	\N
2009	276	\N	\N	1083	embossers cerebrum gloves mohairs admiralties bewigged playgoers cheques batting	\N	\N
394	319	\N	\N	1083	admiralties bewigged playgoers cheques batting waspishly stilbestrol	\N	\N
1899	16	\N	\N	1083	oinked apnea maxillary dumpers bering evasiveness toto	\N	\N
1554	214	\N	\N	1083	recelebrating dually plateauing reoccupations embossers cerebrum gloves mohairs admiralties	\N	\N
398	145	\N	\N	1083	desexes summable heliocentricity excretions recelebrating dually plateauing	\N	\N
415	52	\N	\N	416	desexes summable heliocentricity excretions recelebrating dually plateauing reoccupations embossers	\N	\N
1066	24	\N	\N	416	reaccepts gunneries exorcises pirog desexes summable heliocentricity excretions recelebrating	\N	\N
1087	280	\N	\N	416	batting waspishly stilbestrol villainousness miscalling firefanged skeins	\N	\N
1098	147	\N	\N	416	embossers cerebrum gloves mohairs admiralties bewigged playgoers cheques batting	\N	\N
1065	95	\N	\N	416	recelebrating dually plateauing reoccupations embossers	\N	\N
1063	222	\N	\N	416	admiralties bewigged playgoers cheques batting	\N	\N
1198	72	\N	\N	1392	funnyman commuted lachrymation floweret arcadian	\N	\N
1667	335	\N	\N	1392	grunts venturers valiantly unremorsefully extradites	\N	\N
1616	155	\N	\N	1617	funnyman commuted lachrymation floweret arcadian	\N	\N
1196	300	\N	\N	1392	sodality outcrop slipcases phenylketonuria grunts venturers valiantly unremorsefully extradites	\N	\N
327	186	\N	\N	331	malines cliffhanger entailments reindexed bedstraws thoughtless	\N	\N
1435	366	\N	\N	1436	technicolor plenishes nippy bounden occulters blubberer	\N	\N
1013	365	\N	\N	1844	cooked reformers goodwife technicolor plenishes	\N	\N
618	167	\N	\N	1955	loftily yabber reprovingly blungers dwarflike	\N	\N
1445	29	\N	\N	1568	summable heliocentricity excretions recelebrating dually plateauing reoccupations embossers	\N	\N
1443	227	\N	\N	1568	waspishly stilbestrol villainousness miscalling	\N	\N
1219	75	\N	\N	1092	unrealistic substituting surges preheats loggias reconciliating photocatalyst lenity	\N	\N
126	337	\N	\N	1092	ponderers conditione loathly cancels debiting parrots paraguayans	\N	\N
1090	303	\N	\N	1092	valiantly unremorsefully extradites stollens ponderers conditione	\N	\N
1218	268	\N	\N	1092	slipcases phenylketonuria grunts venturers valiantly unremorsefully extradites stollens ponderers	\N	\N
1695	176	\N	\N	1696	inflecting trephines hops exec junketeers isolators reducing nethermost	\N	\N
1476	281	\N	\N	1477	stilbestrol villainousness miscalling firefanged skeins equalled sandwiching bewitchment	\N	\N
1576	26	\N	\N	1477	exorcises pirog desexes summable heliocentricity	\N	\N
118	80	\N	\N	986	sodality outcrop slipcases phenylketonuria grunts venturers valiantly	\N	\N
119	308	\N	\N	986	paraguayans resonates overbites terminals giros	\N	\N
120	161	\N	\N	986	extradites stollens ponderers conditione loathly	\N	\N
1242	241	\N	\N	1680	splinting surmisers satisfying undamped sharpers forbearer	\N	\N
355	267	\N	\N	2182	outcrop slipcases phenylketonuria grunts venturers valiantly unremorsefully extradites	\N	\N
567	121	\N	\N	1514	householder overeducated objurgate treaties preprocessor despising loftily	\N	\N
1210	56	\N	\N	1211	embossers cerebrum gloves mohairs admiralties bewigged playgoers	\N	\N
401	32	\N	\N	1211	recelebrating dually plateauing reoccupations embossers	\N	\N
1326	284	\N	\N	1211	sandwiching bewitchment cheaters riffled kerneling	\N	\N
402	230	\N	\N	1211	miscalling firefanged skeins equalled sandwiching bewitchment cheaters	\N	\N
294	79	\N	\N	1446	tautological jambing sodality outcrop slipcases phenylketonuria	\N	\N
503	113	\N	\N	1446	slipcases phenylketonuria grunts venturers valiantly	\N	\N
732	35	\N	\N	1072	reoccupations embossers cerebrum gloves mohairs admiralties bewigged playgoers	\N	\N
733	233	\N	\N	1072	equalled sandwiching bewitchment cheaters	\N	\N
811	94	\N	\N	814	summable heliocentricity excretions recelebrating	\N	\N
153	219	\N	\N	814	cerebrum gloves mohairs admiralties bewigged playgoers cheques batting	\N	\N
154	21	\N	\N	814	evasiveness toto teashop reaccepts gunneries exorcises	\N	\N
493	293	\N	\N	446	lachrymation floweret arcadian acridities unrealistic substituting surges preheats	\N	\N
492	248	\N	\N	446	undermentioned outflanking funnyman commuted lachrymation floweret arcadian	\N	\N
1669	263	\N	\N	2038	lenity tautological jambing sodality	\N	\N
310	112	\N	\N	1846	jambing sodality outcrop slipcases	\N	\N
117	114	\N	\N	730	venturers valiantly unremorsefully extradites stollens ponderers	\N	\N
876	358	\N	\N	1342	isolators reducing nethermost nonfiction	\N	\N
159	359	\N	\N	531	nonfiction retrogressions eliminates unknowns mongoloids	\N	\N
1259	175	\N	\N	531	recalculation safest signiory latchets inflecting trephines hops	\N	\N
1372	171	\N	\N	744	bluejays cuppier nonsurgical skimpiest outpoured dissociated heartier petitionee brill	\N	\N
741	127	\N	\N	744	tibetans slavishly hipless prs bluejays cuppier nonsurgical	\N	\N
617	200	\N	\N	1919	emanative blowpipe varies thickest machinability orbiters tormentor owner	\N	\N
371	183	\N	\N	2025	goodwife technicolor plenishes nippy bounden occulters blubberer amenities desecrated	\N	\N
1135	166	\N	\N	2158	objurgate treaties preprocessor despising	\N	\N
521	70	\N	\N	195	forbearer anesthetization undermentioned outflanking funnyman commuted lachrymation floweret arcadian	\N	\N
522	298	\N	\N	195	photocatalyst lenity tautological jambing sodality outcrop slipcases	\N	\N
194	258	\N	\N	195	surges preheats loggias reconciliating photocatalyst	\N	\N
778	156	\N	\N	195	arcadian acridities unrealistic substituting surges preheats	\N	\N
582	107	\N	\N	195	funnyman commuted lachrymation floweret arcadian	\N	\N
387	128	\N	\N	2217	prs bluejays cuppier nonsurgical skimpiest outpoured dissociated heartier	\N	\N
1260	132	\N	\N	2118	fobbed transcribed swifters redigesting ostinato recalculation	\N	\N
319	81	\N	\N	672	slipcases phenylketonuria grunts venturers valiantly unremorsefully extradites stollens	\N	\N
320	309	\N	\N	672	overbites terminals giros podgy vagus kinkiest	\N	\N
761	240	\N	\N	\N	rifer splinting surmisers satisfying undamped	\N	\N
208	229	\N	\N	\N	villainousness miscalling firefanged skeins equalled sandwiching	\N	\N
2175	158	\N	\N	\N	photocatalyst lenity tautological jambing sodality outcrop slipcases phenylketonuria	\N	\N
1068	24	\N	\N	\N	reaccepts gunneries exorcises pirog desexes summable heliocentricity excretions recelebrating	\N	\N
2194	179	\N	\N	\N	mongoloids danker raunchiness perspicuously disjoined	\N	\N
893	45	\N	\N	68	swiveled polers oinked apnea maxillary dumpers bering evasiveness	\N	\N
504	346	\N	\N	507	effulgences coreless tuberculoses environs	\N	\N
1533	153	\N	\N	1308	surmisers satisfying undamped sharpers forbearer anesthetization undermentioned outflanking funnyman	\N	\N
791	331	\N	\N	790	surges preheats loggias reconciliating photocatalyst lenity tautological	\N	\N
1729	224	\N	\N	1477	playgoers cheques batting waspishly stilbestrol villainousness miscalling	\N	\N
653	123	\N	\N	713	loftily yabber reprovingly blungers dwarflike effulgences coreless tuberculoses environs	\N	\N
1325	33	\N	\N	1157	dually plateauing reoccupations embossers cerebrum gloves	\N	\N
123	362	\N	\N	122	nigglings midmonths labium peeped daydreams permuting immediately canzona	\N	\N
422	350	\N	\N	421	cuppier nonsurgical skimpiest outpoured dissociated heartier petitionee brill	\N	\N
921	226	\N	\N	768	batting waspishly stilbestrol villainousness miscalling firefanged skeins equalled sandwiching	\N	\N
105	349	\N	\N	104	hipless prs bluejays cuppier nonsurgical skimpiest outpoured	\N	\N
881	163	\N	\N	224	paraguayans resonates overbites terminals giros podgy vagus	\N	\N
763	117	\N	\N	1399	debiting parrots paraguayans resonates overbites terminals giros podgy vagus	\N	\N
464	23	\N	\N	463	teashop reaccepts gunneries exorcises pirog desexes summable heliocentricity	\N	\N
115	143	\N	\N	110	bering evasiveness toto teashop reaccepts	\N	\N
174	134	\N	\N	815	safest signiory latchets inflecting trephines hops exec junketeers	\N	\N
1252	291	\N	\N	1124	undermentioned outflanking funnyman commuted lachrymation floweret	\N	\N
1440	133	\N	\N	1522	redigesting ostinato recalculation safest signiory latchets inflecting	\N	\N
856	9	\N	\N	789	scrapped prat islanded burlily thanklessly swiveled	\N	\N
1229	343	\N	\N	224	objurgate treaties preprocessor despising loftily yabber reprovingly	\N	\N
1239	182	\N	\N	1238	canzona interrelated cooked reformers goodwife technicolor plenishes nippy	\N	\N
569	283	\N	\N	940	skeins equalled sandwiching bewitchment	\N	\N
265	37	\N	\N	1651	cerebrum gloves mohairs admiralties	\N	\N
1609	255	\N	\N	537	acridities unrealistic substituting surges preheats loggias reconciliating photocatalyst	\N	\N
894	273	\N	\N	68	heliocentricity excretions recelebrating dually plateauing reoccupations	\N	\N
250	357	\N	\N	249	hops exec junketeers isolators reducing nethermost nonfiction retrogressions eliminates	\N	\N
511	4	\N	\N	509	hogfishes gantleted richer succories dwelling scrapped prat	\N	\N
1181	157	\N	\N	1392	surges preheats loggias reconciliating photocatalyst lenity tautological	\N	\N
978	82	\N	\N	1745	grunts venturers valiantly unremorsefully extradites stollens ponderers conditione loathly	\N	\N
562	256	\N	\N	489	unrealistic substituting surges preheats loggias reconciliating photocatalyst lenity tautological	\N	\N
291	364	\N	\N	2017	immediately canzona interrelated cooked	\N	\N
465	221	\N	\N	463	mohairs admiralties bewigged playgoers	\N	\N
454	40	\N	\N	610	admiralties bewigged playgoers cheques batting waspishly stilbestrol	\N	\N
207	197	\N	\N	1709	maladjustments subpartnership cordilleras recirculations alkalin	\N	\N
114	206	\N	\N	110	reaccepts gunneries exorcises pirog desexes summable heliocentricity	\N	\N
29	125	\N	\N	28	coreless tuberculoses environs hulled preexamination	\N	\N
112	44	\N	\N	110	burlily thanklessly swiveled polers oinked apnea maxillary	\N	\N
946	363	\N	\N	108	peeped daydreams permuting immediately canzona interrelated cooked reformers goodwife	\N	\N
1226	229	\N	\N	186	villainousness miscalling firefanged skeins equalled sandwiching	\N	\N
373	355	\N	\N	372	recalculation safest signiory latchets inflecting trephines hops	\N	\N
395	216	\N	\N	32	plateauing reoccupations embossers cerebrum gloves	\N	\N
1534	103	\N	\N	1308	kerneling napoleons rifer splinting surmisers satisfying undamped	\N	\N
922	323	\N	\N	768	sandwiching bewitchment cheaters riffled kerneling	\N	\N
1369	124	\N	\N	1395	blungers dwarflike effulgences coreless	\N	\N
596	185	\N	\N	830	desecrated tetrachlorides loutish polygony malines	\N	\N
308	77	\N	\N	438	loggias reconciliating photocatalyst lenity	\N	\N
1710	339	\N	\N	278	paraguayans resonates overbites terminals giros podgy vagus kinkiest xix	\N	\N
1127	234	\N	\N	14	sandwiching bewitchment cheaters riffled kerneling	\N	\N
1588	321	\N	\N	441	stilbestrol villainousness miscalling firefanged skeins equalled sandwiching bewitchment cheaters	\N	\N
127	76	\N	\N	48	surges preheats loggias reconciliating photocatalyst lenity tautological jambing sodality	\N	\N
1531	292	\N	\N	1308	funnyman commuted lachrymation floweret arcadian acridities unrealistic	\N	\N
381	247	\N	\N	379	anesthetization undermentioned outflanking funnyman commuted lachrymation	\N	\N
147	193	\N	\N	1000	irades inadvisability disciplinarians majors manifestly decaffeinates scalepan	\N	\N
775	294	\N	\N	790	arcadian acridities unrealistic substituting surges preheats loggias reconciliating photocatalyst	\N	\N
282	367	\N	\N	281	bounden occulters blubberer amenities desecrated tetrachlorides loutish	\N	\N
891	46	\N	\N	609	oinked apnea maxillary dumpers bering evasiveness toto teashop reaccepts	\N	\N
141	12	\N	\N	609	burlily thanklessly swiveled polers oinked apnea maxillary dumpers bering	\N	\N
222	312	\N	\N	224	xix recollected householder overeducated objurgate treaties preprocessor despising loftily	\N	\N
1143	177	\N	\N	869	junketeers isolators reducing nethermost nonfiction retrogressions eliminates unknowns mongoloids	\N	\N
865	135	\N	\N	869	inflecting trephines hops exec junketeers isolators reducing nethermost nonfiction	\N	\N
918	290	\N	\N	86	forbearer anesthetization undermentioned outflanking funnyman	\N	\N
81	152	\N	\N	86	kerneling napoleons rifer splinting surmisers satisfying undamped sharpers	\N	\N
1139	311	\N	\N	1221	vagus kinkiest xix recollected householder overeducated objurgate treaties	\N	\N
706	101	\N	\N	377	skeins equalled sandwiching bewitchment cheaters	\N	\N
256	253	\N	\N	392	floweret arcadian acridities unrealistic substituting surges	\N	\N
134	278	\N	\N	1305	admiralties bewigged playgoers cheques batting	\N	\N
90	165	\N	\N	94	xix recollected householder overeducated objurgate treaties preprocessor despising loftily	\N	\N
563	69	\N	\N	489	undamped sharpers forbearer anesthetization undermentioned outflanking funnyman commuted	\N	\N
485	333	\N	\N	489	tautological jambing sodality outcrop slipcases phenylketonuria grunts venturers valiantly	\N	\N
1256	198	\N	\N	1277	alkalin succulently marquise underlaid neurosurgeon innervated	\N	\N
545	180	\N	\N	1333	disjoined nigglings midmonths labium peeped daydreams	\N	\N
60	299	\N	\N	1378	tautological jambing sodality outcrop slipcases phenylketonuria grunts venturers	\N	\N
1377	260	\N	\N	1378	loggias reconciliating photocatalyst lenity tautological jambing sodality	\N	\N
149	86	\N	\N	662	loathly cancels debiting parrots paraguayans resonates overbites	\N	\N
583	164	\N	\N	662	giros podgy vagus kinkiest xix recollected householder overeducated	\N	\N
263	286	\N	\N	14	kerneling napoleons rifer splinting surmisers satisfying undamped	\N	\N
270	150	\N	\N	14	miscalling firefanged skeins equalled sandwiching bewitchment	\N	\N
10	36	\N	\N	14	embossers cerebrum gloves mohairs admiralties bewigged playgoers cheques batting	\N	\N
269	58	\N	\N	14	admiralties bewigged playgoers cheques batting waspishly stilbestrol villainousness miscalling	\N	\N
685	99	\N	\N	14	batting waspishly stilbestrol villainousness miscalling firefanged skeins equalled sandwiching	\N	\N
1404	340	\N	\N	1406	terminals giros podgy vagus	\N	\N
1307	64	\N	\N	1308	sandwiching bewitchment cheaters riffled kerneling napoleons rifer splinting surmisers	\N	\N
1540	201	\N	\N	1544	machinability orbiters tormentor owner zanier corkscrewed promiscuousness clewed reassemble	\N	\N
1069	249	\N	\N	1070	outflanking funnyman commuted lachrymation floweret arcadian acridities unrealistic	\N	\N
299	173	\N	\N	303	brill neologic intermuscular fobbed transcribed	\N	\N
787	207	\N	\N	789	gunneries exorcises pirog desexes summable heliocentricity excretions recelebrating	\N	\N
801	30	\N	\N	940	heliocentricity excretions recelebrating dually plateauing reoccupations embossers cerebrum gloves	\N	\N
802	228	\N	\N	940	stilbestrol villainousness miscalling firefanged skeins	\N	\N
482	122	\N	\N	852	treaties preprocessor despising loftily yabber reprovingly blungers dwarflike	\N	\N
429	14	\N	\N	943	swiveled polers oinked apnea maxillary	\N	\N
624	287	\N	\N	627	rifer splinting surmisers satisfying undamped sharpers forbearer anesthetization	\N	\N
1049	236	\N	\N	627	cheaters riffled kerneling napoleons rifer splinting surmisers	\N	\N
1364	39	\N	\N	1422	mohairs admiralties bewigged playgoers cheques batting	\N	\N
178	245	\N	\N	179	sharpers forbearer anesthetization undermentioned	\N	\N
218	57	\N	\N	165	gloves mohairs admiralties bewigged playgoers cheques batting waspishly	\N	\N
1662	34	\N	\N	165	plateauing reoccupations embossers cerebrum gloves mohairs admiralties	\N	\N
251	325	\N	\N	165	rifer splinting surmisers satisfying undamped sharpers forbearer	\N	\N
160	232	\N	\N	165	skeins equalled sandwiching bewitchment cheaters riffled kerneling napoleons rifer	\N	\N
1398	85	\N	\N	1399	ponderers conditione loathly cancels debiting parrots	\N	\N
1704	189	\N	\N	1705	surrejoinders chestiest measurage tonsils pasturage thurifer teazle fringier rhythmical	\N	\N
211	251	\N	\N	1775	commuted lachrymation floweret arcadian	\N	\N
267	235	\N	\N	1651	bewitchment cheaters riffled kerneling napoleons rifer	\N	\N
903	316	\N	\N	1106	excretions recelebrating dually plateauing	\N	\N
449	60	\N	\N	610	batting waspishly stilbestrol villainousness miscalling	\N	\N
1889	327	\N	\N	610	forbearer anesthetization undermentioned outflanking funnyman commuted lachrymation floweret arcadian	\N	\N
450	151	\N	\N	610	sandwiching bewitchment cheaters riffled kerneling napoleons rifer	\N	\N
456	238	\N	\N	610	kerneling napoleons rifer splinting surmisers satisfying undamped sharpers forbearer	\N	\N
1015	169	\N	\N	1016	environs hulled preexamination oralogy tibetans slavishly hipless	\N	\N
536	106	\N	\N	537	anesthetization undermentioned outflanking funnyman	\N	\N
1883	17	\N	\N	928	apnea maxillary dumpers bering evasiveness toto teashop reaccepts	\N	\N
734	98	\N	\N	1157	bewigged playgoers cheques batting waspishly stilbestrol villainousness miscalling	\N	\N
1156	231	\N	\N	1157	firefanged skeins equalled sandwiching bewitchment cheaters riffled kerneling	\N	\N
1621	192	\N	\N	1800	inserter plights corollaries gaudiness irades inadvisability	\N	\N
400	324	\N	\N	186	riffled kerneling napoleons rifer splinting surmisers	\N	\N
184	31	\N	\N	186	excretions recelebrating dually plateauing	\N	\N
359	302	\N	\N	412	grunts venturers valiantly unremorsefully extradites	\N	\N
408	266	\N	\N	412	sodality outcrop slipcases phenylketonuria grunts venturers valiantly	\N	\N
1250	63	\N	\N	1124	skeins equalled sandwiching bewitchment cheaters riffled kerneling napoleons	\N	\N
1122	329	\N	\N	1124	lachrymation floweret arcadian acridities unrealistic	\N	\N
1173	244	\N	\N	1124	undamped sharpers forbearer anesthetization undermentioned outflanking funnyman commuted lachrymation	\N	\N
1160	92	\N	\N	1878	teashop reaccepts gunneries exorcises pirog desexes summable heliocentricity	\N	\N
1161	213	\N	\N	1878	excretions recelebrating dually plateauing reoccupations embossers cerebrum gloves	\N	\N
1737	269	\N	\N	1738	phenylketonuria grunts venturers valiantly	\N	\N
1352	93	\N	\N	32	exorcises pirog desexes summable heliocentricity excretions recelebrating dually plateauing	\N	\N
30	49	\N	\N	32	toto teashop reaccepts gunneries exorcises pirog	\N	\N
397	18	\N	\N	32	maxillary dumpers bering evasiveness toto teashop reaccepts gunneries exorcises	\N	\N
847	352	\N	\N	1840	petitionee brill neologic intermuscular	\N	\N
\.



COPY "PaperAbstract" ("ID") FROM stdin;
6
7
25
39
65
72
89
99
137
171
176
177
190
193
208
217
231
289
296
313
337
388
403
407
423
439
471
498
519
533
552
577
593
614
620
646
656
681
682
712
728
737
740
751
759
761
786
794
797
810
821
828
829
842
854
862
872
877
884
897
925
936
937
950
954
955
961
966
983
988
999
1001
1014
1047
1048
1059
1067
1068
1078
1105
1126
1136
1148
1165
1172
1179
1193
1202
1207
1230
1237
1245
1261
1274
1283
1285
1292
1293
1299
1304
1312
1314
1321
1322
1332
1346
1349
1358
1370
1381
1388
1391
1396
1397
1411
1413
1429
1448
1458
1463
1474
1480
1490
1495
1499
1500
1505
1510
1516
1521
1524
1525
1527
1535
1552
1563
1565
1570
1574
1575
1583
1584
1590
1602
1604
1610
1611
1622
1627
1652
1653
1656
1658
1659
1663
1664
1675
1678
1684
1690
1703
1708
1717
1728
1735
1740
1747
1753
1758
1766
1772
1777
1781
1782
1783
1785
1789
1790
1793
1796
1797
1806
1811
1815
1817
1820
1822
1827
1829
1832
1833
1835
1841
1843
1848
1850
1853
1857
1862
1865
1871
1876
1877
1879
1880
1888
1891
1893
1897
1907
1909
1911
1914
1925
1927
1929
1930
1934
1935
1941
1946
1948
1949
1956
1965
1968
1971
1972
1974
1975
1977
1978
1980
1982
1988
1995
1999
2002
2003
2005
2011
2013
2015
2018
2020
2023
2024
2027
2033
2036
2039
2040
2041
2044
2048
2049
2051
2055
2058
2062
2067
2072
2074
2076
2077
2078
2082
2084
2086
2088
2091
2096
2098
2099
2103
2104
2107
2108
2109
2110
2111
2112
2114
2116
2120
2121
2122
2123
2127
2133
2134
2139
2140
2141
2143
2144
2147
2148
2153
2155
2156
2159
2162
2163
2164
2166
2167
2168
2172
2174
2175
2178
2179
2181
2183
2184
2186
2187
2189
2191
2192
2193
2194
2195
2197
2199
2200
2201
2202
2203
2205
2207
2208
2211
2214
2215
2216
2218
2220
2221
2222
2224
2225
2227
2228
2230
2231
2233
2234
2235
2236
2237
2238
2240
2241
2242
2243
2244
2246
2248
2249
2250
2251
2252
2253
2254
2255
\.



COPY "PaperFullVersion" ("ID") FROM stdin;
10
19
29
30
37
38
42
43
49
59
60
62
66
81
90
98
105
111
112
113
114
115
117
118
119
120
123
124
126
127
129
132
133
134
140
141
143
147
149
150
153
154
159
160
174
178
184
192
194
207
211
212
218
222
250
251
256
257
259
260
263
264
265
267
269
270
274
282
283
284
291
294
295
299
308
309
310
319
320
327
333
334
335
340
341
345
346
347
349
350
351
353
354
355
356
359
360
371
373
374
375
380
381
387
394
395
396
397
398
400
401
402
405
408
415
422
429
432
433
435
436
440
443
449
450
453
454
456
457
458
464
465
474
482
485
492
493
503
504
510
511
512
514
521
522
536
545
558
559
562
563
567
569
571
575
581
582
583
592
596
606
617
618
623
624
628
653
659
669
685
689
694
705
706
714
727
732
733
734
741
754
755
756
763
765
771
773
775
776
778
781
785
787
791
801
802
806
811
824
827
832
833
847
856
857
864
865
876
881
883
888
891
893
894
895
896
903
906
913
914
915
918
921
922
923
924
926
946
947
949
965
969
976
978
979
991
1010
1013
1015
1021
1029
1030
1032
1038
1049
1051
1057
1058
1063
1065
1066
1069
1082
1087
1090
1098
1099
1111
1117
1122
1127
1135
1139
1140
1143
1151
1156
1159
1160
1161
1173
1181
1182
1184
1188
1192
1196
1197
1198
1200
1210
1218
1219
1225
1226
1229
1239
1242
1243
1244
1250
1252
1256
1259
1260
1268
1272
1281
1298
1307
1317
1324
1325
1326
1352
1359
1363
1364
1366
1369
1372
1377
1384
1385
1390
1398
1404
1435
1440
1443
1445
1450
1476
1483
1484
1518
1531
1532
1533
1534
1539
1540
1554
1557
1558
1559
1576
1585
1587
1588
1596
1609
1616
1619
1620
1621
1631
1646
1647
1662
1667
1669
1695
1704
1710
1714
1729
1730
1737
1751
1757
1771
1883
1889
1899
2009
2102
\.



COPY "Person" ("ID", name, email) FROM stdin;
2129	Nessa Thomison	Nessa.Thomison@secretresearch.tld
766	Shaakira Ricklefs	Shaakira.Ricklefs@secretresearch.tld
1376	Che Kess	Che.Kess@secretresearch.tld
301	Trillare Stukel	Trillare.Stukel@secretresearch.tld
793	Argus Bierwirth	Argus.Bierwirth@secretresearch.tld
391	Takako Nordine	Takako.Nordine@secretresearch.tld
1089	Cenon Colon	Cenon.Colon@secretresearch.tld
525	Lili Atoe	Lili.Atoe@secretresearch.tld
35	Shesha Humbird	Shesha.Humbird@secretresearch.tld
138	Torah Giller	Torah.Giller@secretresearch.tld
209	JamAan Kine	JamAan.Kine@secretresearch.tld
279	Deelon Sephton	Deelon.Sephton@secretresearch.tld
316	Dijana Murfin	Dijana.Murfin@secretresearch.tld
315	Evamarie Coraham	Evamarie.Coraham@secretresearch.tld
197	Lance Fernandes	Lance.Fernandes@secretresearch.tld
556	Fedoritsa Stinner	Fedoritsa.Stinner@secretresearch.tld
573	Immanuel Bugos	Immanuel.Bugos@secretresearch.tld
579	Eshita Whitby	Eshita.Whitby@secretresearch.tld
666	Kaveri Zombro	Kaveri.Zombro@secretresearch.tld
703	Radoste Hoefling	Radoste.Hoefling@secretresearch.tld
725	Akina Jang	Akina.Jang@secretresearch.tld
803	Asami Sellek	Asami.Sellek@secretresearch.tld
107	Krystal Farria	Krystal.Farria@secretresearch.tld
590	Naglaya Tantillo	Naglaya.Tantillo@secretresearch.tld
92	Katreen Osthoff	Katreen.Osthoff@secretresearch.tld
300	Yoriko Boot	Yoriko.Boot@secretresearch.tld
676	Nolan Brownle	Nolan.Brownle@secretresearch.tld
1132	Dirthe Kiewiet	Dirthe.Kiewiet@secretresearch.tld
241	Emina Yeeloy	Emina.Yeeloy@secretresearch.tld
686	Mikolas Sirman	Mikolas.Sirman@secretresearch.tld
1494	Mawar Camfield	Mawar.Camfield@secretresearch.tld
181	Aristomnis Dauber	Aristomnis.Dauber@secretresearch.tld
91	Miloushka Pezez	Miloushka.Pezez@secretresearch.tld
687	Flaca Sodawasser	Flaca.Sodawasser@secretresearch.tld
1295	Luitfriede Westmark	Luitfriede.Westmark@secretresearch.tld
1764	Daniel Harriston	Daniel.Harriston@secretresearch.tld
298	Odysseus Dilthey	Odysseus.Dilthey@secretresearch.tld
708	Garon Tischler	Garon.Tischler@secretresearch.tld
1635	Nisus Coby	Nisus.Coby@secretresearch.tld
486	Lucile Homola	Lucile.Homola@secretresearch.tld
975	Helene Nik	Helene.Nik@secretresearch.tld
71	Alexandros Lagazo	Alexandros.Lagazo@secretresearch.tld
51	Martina Mcclary	Martina.Mcclary@secretresearch.tld
528	Iasion Kirt	Iasion.Kirt@secretresearch.tld
1270	Iudita Altew	Iudita.Altew@secretresearch.tld
1687	Nana Stania	Nana.Stania@secretresearch.tld
1334	Ayla Moncrief	Ayla.Moncrief@secretresearch.tld
1905	Duron Dibella	Duron.Dibella@secretresearch.tld
707	Kim Colombe	Kim.Colombe@secretresearch.tld
1636	Cyril Croslen	Cyril.Croslen@secretresearch.tld
1767	Cleveland Navarez	Cleveland.Navarez@secretresearch.tld
933	Malcolm Voelz	Malcolm.Voelz@secretresearch.tld
825	Erminia Damrell	Erminia.Damrell@secretresearch.tld
96	Muni Mccuiston	Muni.Mccuiston@secretresearch.tld
837	Daris Astley	Daris.Astley@secretresearch.tld
2017	Mudri Gamero	Mudri.Gamero@secretresearch.tld
1761	Madelynn Hoehn	Madelynn.Hoehn@secretresearch.tld
1236	Editta Servoss	Editta.Servoss@secretresearch.tld
421	Glafira Geng	Glafira.Geng@secretresearch.tld
1844	Sameera Beazley	Sameera.Beazley@secretresearch.tld
412	Jurgisr Arkins	Jurgisr.Arkins@secretresearch.tld
1418	Nene Rhea	Nene.Rhea@secretresearch.tld
32	Jelawat Balletta	Jelawat.Balletta@secretresearch.tld
610	LeVaughan Laity	LeVaughan.Laity@secretresearch.tld
1654	Sadahige Taglialatela	Sadahige.Taglialatela@secretresearch.tld
1211	Takehide Gauna	Takehide.Gauna@secretresearch.tld
489	Hanka Henness	Hanka.Henness@secretresearch.tld
1618	Anwar Katayama	Anwar.Katayama@secretresearch.tld
1507	Inocente Arntzen	Inocente.Arntzen@secretresearch.tld
1125	Yoshida Volo	Yoshida.Volo@secretresearch.tld
1553	Shandee Finnicum	Shandee.Finnicum@secretresearch.tld
796	Cornelius Dolak	Cornelius.Dolak@secretresearch.tld
461	Kuemon Coatie	Kuemon.Coatie@secretresearch.tld
2182	Lippio Llopis	Lippio.Llopis@secretresearch.tld
852	Dodie Tedford	Dodie.Tedford@secretresearch.tld
1680	Radha Sloan	Radha.Sloan@secretresearch.tld
630	Gota Dorff	Gota.Dorff@secretresearch.tld
1955	Khynika Borwig	Khynika.Borwig@secretresearch.tld
1333	Namiko Leval	Namiko.Leval@secretresearch.tld
1449	Ruwaid Solian	Ruwaid.Solian@secretresearch.tld
789	Eupeithes Bevens	Eupeithes.Bevens@secretresearch.tld
744	Flora Aveni	Flora.Aveni@secretresearch.tld
348	Joey Yarboro	Joey.Yarboro@secretresearch.tld
2056	Junichi Hagele	Junichi.Hagele@secretresearch.tld
165	Boje Ostrum	Boje.Ostrum@secretresearch.tld
662	Pippo Nascimento	Pippo.Nascimento@secretresearch.tld
588	Zufar Pestoni	Zufar.Pestoni@secretresearch.tld
278	Fikriyya Siddiq	Fikriyya.Siddiq@secretresearch.tld
110	Yemena Knebel	Yemena.Knebel@secretresearch.tld
830	Oles Sumaran	Oles.Sumaran@secretresearch.tld
943	Gulika Alipio	Gulika.Alipio@secretresearch.tld
186	Khadija Lagerquist	Khadija.Lagerquist@secretresearch.tld
713	Abdul Spicer	Abdul.Spicer@secretresearch.tld
907	Lauranne Brwon	Lauranne.Brwon@secretresearch.tld
2124	Stopolcha Starowicz	Stopolcha.Starowicz@secretresearch.tld
591	Fabiana Demase	Fabiana.Demase@secretresearch.tld
1562	Querida Kreatsoulas	Querida.Kreatsoulas@secretresearch.tld
831	Caden Failing	Caden.Failing@secretresearch.tld
1016	Tuhfa Pedez	Tuhfa.Pedez@secretresearch.tld
1092	Podolphe Fluitt	Podolphe.Fluitt@secretresearch.tld
2016	Bhavin Primavera	Bhavin.Primavera@secretresearch.tld
1276	Waliyya Dandrea	Waliyya.Dandrea@secretresearch.tld
1036	Janita Bolieu	Janita.Bolieu@secretresearch.tld
1362	Graud Shuker	Graud.Shuker@secretresearch.tld
885	Hortense Ganotisi	Hortense.Ganotisi@secretresearch.tld
981	Everardo Yidiaris	Everardo.Yidiaris@secretresearch.tld
1845	Liebetraud Kiddy	Liebetraud.Kiddy@secretresearch.tld
1116	Amora Ockmond	Amora.Ockmond@secretresearch.tld
1331	Jakoba Bueche	Jakoba.Bueche@secretresearch.tld
1642	Salaah Elio	Salaah.Elio@secretresearch.tld
1910	Dele Galla	Dele.Galla@secretresearch.tld
1882	Hedda Yoshida	Hedda.Yoshida@secretresearch.tld
1542	Vonda Jasmine	Vonda.Jasmine@secretresearch.tld
1515	Chesell Vatter	Chesell.Vatter@secretresearch.tld
329	Pieter Frend	Pieter.Frend@secretresearch.tld
2090	Giselher Nibler	Giselher.Nibler@secretresearch.tld
1357	Noburo Charleton	Noburo.Charleton@secretresearch.tld
994	Lone Pavese	Lone.Pavese@secretresearch.tld
2210	Natasia Sliter	Natasia.Sliter@secretresearch.tld
1992	Freda Freimuth	Freda.Freimuth@secretresearch.tld
1327	Janah Moryl	Janah.Moryl@secretresearch.tld
1847	Kobe Guttirez	Kobe.Guttirez@secretresearch.tld
1503	Jaen Kerechanko	Jaen.Kerechanko@secretresearch.tld
191	Rhea Platas	Rhea.Platas@secretresearch.tld
1839	Xuqiu Deller	Xuqiu.Deller@secretresearch.tld
1944	Muna Lastovica	Muna.Lastovica@secretresearch.tld
476	Dyllan Mairs	Dyllan.Mairs@secretresearch.tld
566	Gabriele Darsey	Gabriele.Darsey@secretresearch.tld
1343	Iulia Morra	Iulia.Morra@secretresearch.tld
901	Aseel Waddups	Aseel.Waddups@secretresearch.tld
2001	Raktavira Danforth	Raktavira.Danforth@secretresearch.tld
911	Avniel Debiase	Avniel.Debiase@secretresearch.tld
1571	Rebeca Aday	Rebeca.Aday@secretresearch.tld
1733	Imelde Ballin	Imelde.Ballin@secretresearch.tld
688	Plakida Gothe	Plakida.Gothe@secretresearch.tld
1904	Yves Hunsucker	Yves.Hunsucker@secretresearch.tld
745	Breonia Gierling	Breonia.Gierling@secretresearch.tld
604	Myshka Krishnamurthy	Myshka.Krishnamurthy@secretresearch.tld
1605	Sharise Heagy	Sharise.Heagy@secretresearch.tld
597	Evgnios Feasel	Evgnios.Feasel@secretresearch.tld
1062	Yumiko Ginther	Yumiko.Ginther@secretresearch.tld
1222	Gil Holben	Gil.Holben@secretresearch.tld
361	Tsuruki Hazelrigg	Tsuruki.Hazelrigg@secretresearch.tld
1763	Clariee Martucci	Clariee.Martucci@secretresearch.tld
11	Tsumemasa Lloyd	Tsumemasa.Lloyd@secretresearch.tld
867	Gerald Fieselman	Gerald.Fieselman@secretresearch.tld
1561	Garca Diers	Garca.Diers@secretresearch.tld
1042	Bozena Tilly	Bozena.Tilly@secretresearch.tld
477	Yakamochi Alonza	Yakamochi.Alonza@secretresearch.tld
376	Saalim Velasquez	Saalim.Velasquez@secretresearch.tld
1861	Oryan Mirles	Oryan.Mirles@secretresearch.tld
1706	Annunziata Ravi	Annunziata.Ravi@secretresearch.tld
1355	Marlene Dague	Marlene.Dague@secretresearch.tld
1801	Lilly Rissanen	Lilly.Rissanen@secretresearch.tld
912	Fajyaz Mutana	Fajyaz.Mutana@secretresearch.tld
1425	Goncalo Gahm	Goncalo.Gahm@secretresearch.tld
1629	Liebtraud Schorsch	Liebtraud.Schorsch@secretresearch.tld
1023	Lyron Strawder	Lyron.Strawder@secretresearch.tld
929	Cicely Genna	Cicely.Genna@secretresearch.tld
205	Rodell Sonnefeld	Rodell.Sonnefeld@secretresearch.tld
234	Elvera Zito	Elvera.Zito@secretresearch.tld
1938	Crysta Mateen	Crysta.Mateen@secretresearch.tld
304	Berakhiah Province	Berakhiah.Province@secretresearch.tld
479	Alfonso Billen	Alfonso.Billen@secretresearch.tld
1154	Melker Kayastha	Melker.Kayastha@secretresearch.tld
1054	Bailee Chabolla	Bailee.Chabolla@secretresearch.tld
1360	Teana Stallworth	Teana.Stallworth@secretresearch.tld
621	Aracely Kuka	Aracely.Kuka@secretresearch.tld
1492	Holda Dible	Holda.Dible@secretresearch.tld
549	Eberhardine Hazim	Eberhardine.Hazim@secretresearch.tld
568	Masakazu Lisko	Masakazu.Lisko@secretresearch.tld
469	ris Kotter	ris.Kotter@secretresearch.tld
973	Enya Coll	Enya.Coll@secretresearch.tld
874	Sophronia Peakes	Sophronia.Peakes@secretresearch.tld
1725	Feofana Knudsen	Feofana.Knudsen@secretresearch.tld
1019	Hita Porell	Hita.Porell@secretresearch.tld
724	Aonghas Pashley	Aonghas.Pashley@secretresearch.tld
718	Ariela Harpine	Ariela.Harpine@secretresearch.tld
1481	Satoru Beaumont	Satoru.Beaumont@secretresearch.tld
941	Vedetta Pilkins	Vedetta.Pilkins@secretresearch.tld
967	Lex Hoven	Lex.Hoven@secretresearch.tld
1190	Naazil Buchannan	Naazil.Buchannan@secretresearch.tld
1262	Dandrae Dorne	Dandrae.Dorne@secretresearch.tld
721	Wolfhild Torros	Wolfhild.Torros@secretresearch.tld
647	Kaspar Reierson	Kaspar.Reierson@secretresearch.tld
312	Anissa Sciarretta	Anissa.Sciarretta@secretresearch.tld
483	Hortencia Brenda	Hortencia.Brenda@secretresearch.tld
1060	Fjodora Sleger	Fjodora.Sleger@secretresearch.tld
1537	Kemina Wyke	Kemina.Wyke@secretresearch.tld
690	Alonza Galyan	Alonza.Galyan@secretresearch.tld
516	Bruno Barthe	Bruno.Barthe@secretresearch.tld
1699	Severin Buzard	Severin.Buzard@secretresearch.tld
716	Sibylla Reas	Sibylla.Reas@secretresearch.tld
1637	Leopoldina Alvey	Leopoldina.Alvey@secretresearch.tld
1141	Vasuki Riech	Vasuki.Riech@secretresearch.tld
1543	Traci Erazmus	Traci.Erazmus@secretresearch.tld
109	Becca Curney	Becca.Curney@secretresearch.tld
1442	Daya Luhr	Daya.Luhr@secretresearch.tld
324	Guiletta Rosek	Guiletta.Rosek@secretresearch.tld
285	Husain Coaxum	Husain.Coaxum@secretresearch.tld
615	Salmoneus Duncans	Salmoneus.Duncans@secretresearch.tld
548	Nandin Uribazo	Nandin.Uribazo@secretresearch.tld
1041	Vrata Socks	Vrata.Socks@secretresearch.tld
908	Harumi Valenza	Harumi.Valenza@secretresearch.tld
1437	Clarinda Oberg	Clarinda.Oberg@secretresearch.tld
555	Krispin Macnamee	Krispin.Macnamee@secretresearch.tld
1452	Irmgard Accala	Irmgard.Accala@secretresearch.tld
1633	Agamemnon Bouras	Agamemnon.Bouras@secretresearch.tld
1169	Tashiya Marrable	Tashiya.Marrable@secretresearch.tld
723	Ingetrud Balezentis	Ingetrud.Balezentis@secretresearch.tld
74	Kantha Lybert	Kantha.Lybert@secretresearch.tld
40	Edmondo Rommelfanger	Edmondo.Rommelfanger@secretresearch.tld
1748	Ekantika Bisogno	Ekantika.Bisogno@secretresearch.tld
850	Zanthe Felarca	Zanthe.Felarca@secretresearch.tld
233	Anbessa Gulde	Anbessa.Gulde@secretresearch.tld
942	Solange Alson	Solange.Alson@secretresearch.tld
1018	Misty Holladay	Misty.Holladay@secretresearch.tld
488	Leonora Peli	Leonora.Peli@secretresearch.tld
302	Huaning Zwolski	Huaning.Zwolski@secretresearch.tld
1231	Atsumichi Licano	Atsumichi.Licano@secretresearch.tld
1424	Calida Starace	Calida.Starace@secretresearch.tld
410	Reiko Castile	Reiko.Castile@secretresearch.tld
1279	Gia Stonesifer	Gia.Stonesifer@secretresearch.tld
411	Libera Ahlborn	Libera.Ahlborn@secretresearch.tld
1555	Toku Synowiec	Toku.Synowiec@secretresearch.tld
625	Lloyd Stallsmith	Lloyd.Stallsmith@secretresearch.tld
1175	Hidemichi Tropp	Hidemichi.Tropp@secretresearch.tld
1123	Handy Ratajczak	Handy.Ratajczak@secretresearch.tld
1120	Deke Mirabelli	Deke.Mirabelli@secretresearch.tld
31	Olzhbeta Hasenberg	Olzhbeta.Hasenberg@secretresearch.tld
78	Anthony Row	Anthony.Row@secretresearch.tld
541	Eskil Littledave	Eskil.Littledave@secretresearch.tld
1034	Casswallawn Scheiblich	Casswallawn.Scheiblich@secretresearch.tld
379	Mukhtaar Grewe	Mukhtaar.Grewe@secretresearch.tld
986	Sabina Conde	Sabina.Conde@secretresearch.tld
86	Addison Bovio	Addison.Bovio@secretresearch.tld
2073	Ayushmati Appelgren	Ayushmati.Appelgren@secretresearch.tld
104	Llwyr Fagerstrom	Llwyr.Fagerstrom@secretresearch.tld
2212	Mimi Stremmel	Mimi.Stremmel@secretresearch.tld
2050	Fahroni Anglea	Fahroni.Anglea@secretresearch.tld
28	Nomi Rosato	Nomi.Rosato@secretresearch.tld
1544	Cipriana Wadzinski	Cipriana.Wadzinski@secretresearch.tld
1446	Juro Scogin	Juro.Scogin@secretresearch.tld
1000	Apelino Reeber	Apelino.Reeber@secretresearch.tld
1337	Diana Gulbranson	Diana.Gulbranson@secretresearch.tld
531	Baptist Schubert	Baptist.Schubert@secretresearch.tld
249	Julika Smar	Julika.Smar@secretresearch.tld
446	Kallie Bednarek	Kallie.Bednarek@secretresearch.tld
1308	Kulthum Liverance	Kulthum.Liverance@secretresearch.tld
1800	Mercede Naslund	Mercede.Naslund@secretresearch.tld
695	Utathya Kreig	Utathya.Kreig@secretresearch.tld
2118	Petros Josefy	Petros.Josefy@secretresearch.tld
1473	Zeenab Lasane	Zeenab.Lasane@secretresearch.tld
940	Gulielma Arce	Gulielma.Arce@secretresearch.tld
1100	Sela Funnye	Sela.Funnye@secretresearch.tld
1994	AbdulBaasid Chilvers	AbdulBaasid.Chilvers@secretresearch.tld
2025	Evdxios Fageraes	Evdxios.Fageraes@secretresearch.tld
1070	Vyshia Waldall	Vyshia.Waldall@secretresearch.tld
438	Ran Rubendall	Ran.Rubendall@secretresearch.tld
1189	Shaunene Libera	Shaunene.Libera@secretresearch.tld
1124	Dorinda Cuchiara	Dorinda.Cuchiara@secretresearch.tld
1651	Isha Quave	Isha.Quave@secretresearch.tld
1477	Anzhela Venturelli	Anzhela.Venturelli@secretresearch.tld
1392	Kiraanna Scherping	Kiraanna.Scherping@secretresearch.tld
1191	Fedyenka Terrasi	Fedyenka.Terrasi@secretresearch.tld
463	Doppo Kieser	Doppo.Kieser@secretresearch.tld
2158	Constantino Thielen	Constantino.Thielen@secretresearch.tld
303	Dukinea Zmijewski	Dukinea.Zmijewski@secretresearch.tld
1705	Bhadra Whidden	Bhadra.Whidden@secretresearch.tld
869	Ajanta Mischel	Ajanta.Mischel@secretresearch.tld
2038	Tarrant Alsbrooks	Tarrant.Alsbrooks@secretresearch.tld
1738	Marulia Bondura	Marulia.Bondura@secretresearch.tld
1180	Finn Carboneau	Finn.Carboneau@secretresearch.tld
68	Luzia Rahib	Luzia.Rahib@secretresearch.tld
1709	Ovidio Bostrom	Ovidio.Bostrom@secretresearch.tld
1522	Carevel Tardif	Carevel.Tardif@secretresearch.tld
195	Strezislava Weingard	Strezislava.Weingard@secretresearch.tld
672	Jedrick Relacion	Jedrick.Relacion@secretresearch.tld
52	Cecil Kochler	Cecil.Kochler@secretresearch.tld
1399	Paloma Schnall	Paloma.Schnall@secretresearch.tld
108	Jared Torgersen	Jared.Torgersen@secretresearch.tld
1617	Augusto Kempson	Augusto.Kempson@secretresearch.tld
1009	Meena Riegel	Meena.Riegel@secretresearch.tld
1517	Chantrelle Gruiger	Chantrelle.Gruiger@secretresearch.tld
224	Yoyon Lecroy	Yoyon.Lecroy@secretresearch.tld
179	Nia Drakes	Nia.Drakes@secretresearch.tld
372	Winand Poulton	Winand.Poulton@secretresearch.tld
392	Masafumi Gipson	Masafumi.Gipson@secretresearch.tld
1342	Patrido Sielaff	Patrido.Sielaff@secretresearch.tld
790	Mazcho Stringer	Mazcho.Stringer@secretresearch.tld
964	Stolpolcha Shigemi	Stolpolcha.Shigemi@secretresearch.tld
1568	Egor Braue	Egor.Braue@secretresearch.tld
57	Michizane Krips	Michizane.Krips@secretresearch.tld
116	Aspasa Abaunza	Aspasa.Abaunza@secretresearch.tld
47	Carli Maggert	Carli.Maggert@secretresearch.tld
128	Tarran Pointer	Tarran.Pointer@secretresearch.tld
151	Hilary Braddy	Hilary.Braddy@secretresearch.tld
255	Maysak Anetsberger	Maysak.Anetsberger@secretresearch.tld
262	Irminwin Milloy	Irminwin.Milloy@secretresearch.tld
352	Dondrea Crocco	Dondrea.Crocco@secretresearch.tld
16	Ofelia Golde	Ofelia.Golde@secretresearch.tld
426	Tydeus Dyckman	Tydeus.Dyckman@secretresearch.tld
448	Toyoko Colden	Toyoko.Colden@secretresearch.tld
445	Koshiro Augusto	Koshiro.Augusto@secretresearch.tld
539	Mora Christofferse	Mora.Christofferse@secretresearch.tld
500	Silvester Yelvington	Silvester.Yelvington@secretresearch.tld
230	Richild Brinius	Richild.Brinius@secretresearch.tld
769	Deverell Anez	Deverell.Anez@secretresearch.tld
777	Jazmine Susman	Jazmine.Susman@secretresearch.tld
46	Faith Neeld	Faith.Neeld@secretresearch.tld
200	Lane Portes	Lane.Portes@secretresearch.tld
607	Guido Genier	Guido.Genier@secretresearch.tld
201	Booker Spiker	Booker.Spiker@secretresearch.tld
643	Angha Bievenue	Angha.Bievenue@secretresearch.tld
768	Bia Klemme	Bia.Klemme@secretresearch.tld
1	Shinji Kapler	Shinji.Kapler@secretresearch.tld
254	Katyushka Gabriele	Katyushka.Gabriele@secretresearch.tld
247	Shaqeeqa Levra	Shaqeeqa.Levra@secretresearch.tld
409	Soshu Burress	Soshu.Burress@secretresearch.tld
85	Alyss Wuitschick	Alyss.Wuitschick@secretresearch.tld
216	Agpios Bia	Agpios.Bia@secretresearch.tld
417	Jenna Ovit	Jenna.Ovit@secretresearch.tld
13	Dymas Houskeeper	Dymas.Houskeeper@secretresearch.tld
1137	Athindoros Shurgot	Athindoros.Shurgot@secretresearch.tld
183	Ieshige Kosinski	Ieshige.Kosinski@secretresearch.tld
244	Demont Escalet	Demont.Escalet@secretresearch.tld
326	Karina Eggington	Karina.Eggington@secretresearch.tld
1194	Seriozhenka Homen	Seriozhenka.Homen@secretresearch.tld
1091	Tadao Pentaris	Tadao.Pentaris@secretresearch.tld
609	Dena Mcentire	Dena.Mcentire@secretresearch.tld
1240	Giovanna Douthit	Giovanna.Douthit@secretresearch.tld
275	Yaineris Francisco	Yaineris.Francisco@secretresearch.tld
157	Kiuprila Inscore	Kiuprila.Inscore@secretresearch.tld
750	Einian Frascella	Einian.Frascella@secretresearch.tld
1351	Maureo Benedict	Maureo.Benedict@secretresearch.tld
613	Kristin Sandhu	Kristin.Sandhu@secretresearch.tld
684	Valentina Topia	Valentina.Topia@secretresearch.tld
45	Sypovaia Wegmann	Sypovaia.Wegmann@secretresearch.tld
1387	Blanca Ursery	Blanca.Ursery@secretresearch.tld
995	Firdaus Casavez	Firdaus.Casavez@secretresearch.tld
367	Octave Diana	Octave.Diana@secretresearch.tld
88	Charlette Negron	Charlette.Negron@secretresearch.tld
14	Dawn Lauinger	Dawn.Lauinger@secretresearch.tld
631	Coireall Shelor	Coireall.Shelor@secretresearch.tld
634	Aetos Bitar	Aetos.Bitar@secretresearch.tld
1052	Notburg Towber	Notburg.Towber@secretresearch.tld
273	Karita Sukovich	Karita.Sukovich@secretresearch.tld
709	Erik Strothmann	Erik.Strothmann@secretresearch.tld
245	Caddock Stagowski	Caddock.Stagowski@secretresearch.tld
1003	Inez Meeroff	Inez.Meeroff@secretresearch.tld
1199	Nik Reposa	Nik.Reposa@secretresearch.tld
1050	Lottchen Pillitteri	Lottchen.Pillitteri@secretresearch.tld
636	Bhoomika Secord	Bhoomika.Secord@secretresearch.tld
475	Elli Asner	Elli.Asner@secretresearch.tld
203	Gustel Pitek	Gustel.Pitek@secretresearch.tld
84	Cindi Ginanni	Cindi.Ginanni@secretresearch.tld
812	Makhna Stabile	Makhna.Stabile@secretresearch.tld
163	Gratia Flach	Gratia.Flach@secretresearch.tld
795	Eileen Heick	Eileen.Heick@secretresearch.tld
993	Motoyasu Calligy	Motoyasu.Calligy@secretresearch.tld
642	Dores Hugi	Dores.Hugi@secretresearch.tld
444	Teriuihi Lachner	Teriuihi.Lachner@secretresearch.tld
189	Davida Rho	Davida.Rho@secretresearch.tld
242	Shiko Seagroves	Shiko.Seagroves@secretresearch.tld
788	Vasilista Hamic	Vasilista.Hamic@secretresearch.tld
214	Effie Menzie	Effie.Menzie@secretresearch.tld
1297	Coby Smaller	Coby.Smaller@secretresearch.tld
640	Tienne Segraves	Tienne.Segraves@secretresearch.tld
1380	Dewi Rappleyea	Dewi.Rappleyea@secretresearch.tld
442	Lexi Faulk	Lexi.Faulk@secretresearch.tld
538	Uleia Mineah	Uleia.Mineah@secretresearch.tld
644	Vseslava Szollosi	Vseslava.Szollosi@secretresearch.tld
419	Atsumori Bloyer	Atsumori.Bloyer@secretresearch.tld
632	Cathy Phebus	Cathy.Phebus@secretresearch.tld
633	Hasnaa Sietsema	Hasnaa.Sietsema@secretresearch.tld
626	Gavriella Sasson	Gavriella.Sasson@secretresearch.tld
1205	Colombe Bermea	Colombe.Bermea@secretresearch.tld
1080	Warona Pelligrino	Warona.Pelligrino@secretresearch.tld
288	Adriaan Belter	Adriaan.Belter@secretresearch.tld
645	Reece Raburn	Reece.Raburn@secretresearch.tld
927	Gorshedna Marone	Gorshedna.Marone@secretresearch.tld
83	Mikio Brunston	Mikio.Brunston@secretresearch.tld
1671	Juanisha Seppi	Juanisha.Seppi@secretresearch.tld
44	Vladlena Tlumacki	Vladlena.Tlumacki@secretresearch.tld
661	Uchdryd Locicero	Uchdryd.Locicero@secretresearch.tld
743	Hypnos Blank	Hypnos.Blank@secretresearch.tld
447	Gizla Nordmark	Gizla.Nordmark@secretresearch.tld
957	Arcadia Granade	Arcadia.Granade@secretresearch.tld
639	Bezubaia Wiece	Bezubaia.Wiece@secretresearch.tld
238	Sander Garriss	Sander.Garriss@secretresearch.tld
369	Dianmu Aver	Dianmu.Aver@secretresearch.tld
1423	Chebi Selking	Chebi.Selking@secretresearch.tld
76	Ineng Alkbsh	Ineng.Alkbsh@secretresearch.tld
223	Zarek Laasaga	Zarek.Laasaga@secretresearch.tld
162	Socorro Mcsorley	Socorro.Mcsorley@secretresearch.tld
638	Aleydis Virgadamo	Aleydis.Virgadamo@secretresearch.tld
1674	Lucrecia Rollings	Lucrecia.Rollings@secretresearch.tld
33	Fedorav Solich	Fedorav.Solich@secretresearch.tld
739	Alona Lucario	Alona.Lucario@secretresearch.tld
240	Wava Goubeaux	Wava.Goubeaux@secretresearch.tld
12	Patroklus Hertel	Patroklus.Hertel@secretresearch.tld
460	Letizia Madaras	Letizia.Madaras@secretresearch.tld
187	Kazushige Goeppner	Kazushige.Goeppner@secretresearch.tld
1083	Timun Silverstein	Timun.Silverstein@secretresearch.tld
416	Juri Guzik	Juri.Guzik@secretresearch.tld
3	Dama Leino	Dama.Leino@secretresearch.tld
601	Sasson Kniffin	Sasson.Kniffin@secretresearch.tld
161	Wira Majette	Wira.Majette@secretresearch.tld
232	Kit Storrs	Kit.Storrs@secretresearch.tld
228	Bodil Korinta	Bodil.Korinta@secretresearch.tld
199	Miranda Leinen	Miranda.Leinen@secretresearch.tld
1008	Shontal Sehnert	Shontal.Sehnert@secretresearch.tld
599	Trina Sjerven	Trina.Sjerven@secretresearch.tld
611	Houke Luthe	Houke.Luthe@secretresearch.tld
956	Chasidy Hoop	Chasidy.Hoop@secretresearch.tld
1400	Nonna Laplaunt	Nonna.Laplaunt@secretresearch.tld
414	Tyonna Sturdevant	Tyonna.Sturdevant@secretresearch.tld
219	Adrien Hoit	Adrien.Hoit@secretresearch.tld
239	Juriaan Schremp	Juriaan.Schremp@secretresearch.tld
420	Aandranee Sakamaki	Aandranee.Sakamaki@secretresearch.tld
80	Tatiiana Sublett	Tatiiana.Sublett@secretresearch.tld
22	Kellie Frenner	Kellie.Frenner@secretresearch.tld
608	Gennadi Schanck	Gennadi.Schanck@secretresearch.tld
919	Grite Faddis	Grite.Faddis@secretresearch.tld
535	Raaid Chrispen	Raaid.Chrispen@secretresearch.tld
487	Llyn Hardge	Llyn.Hardge@secretresearch.tld
248	Oka Prock	Oka.Prock@secretresearch.tld
473	Ernestine Barkan	Ernestine.Barkan@secretresearch.tld
390	Loreliese Aspri	Loreliese.Aspri@secretresearch.tld
1577	Erechtheus Muschick	Erechtheus.Muschick@secretresearch.tld
103	Melisa Patriarco	Melisa.Patriarco@secretresearch.tld
637	Raimund Widdowson	Raimund.Widdowson@secretresearch.tld
673	Francoise Rioux	Francoise.Rioux@secretresearch.tld
276	Kakuei Blanford	Kakuei.Blanford@secretresearch.tld
393	Brittani Gasman	Brittani.Gasman@secretresearch.tld
246	Breana Scheffler	Breana.Scheffler@secretresearch.tld
1094	Conchetta Cater	Conchetta.Cater@secretresearch.tld
164	Cassian Cozzy	Cassian.Cozzy@secretresearch.tld
24	Uqbah Oconnell	Uqbah.Oconnell@secretresearch.tld
237	Lene Vosquez	Lene.Vosquez@secretresearch.tld
1093	Lucretia Ferndez	Lucretia.Ferndez@secretresearch.tld
1167	Graziana Callabrass	Graziana.Callabrass@secretresearch.tld
813	Loke Tuton	Loke.Tuton@secretresearch.tld
50	Tatsukichi Gerstenberger	Tatsukichi.Gerstenberger@secretresearch.tld
418	Elif Gameros	Elif.Gameros@secretresearch.tld
1112	Lilja Siwiec	Lilja.Siwiec@secretresearch.tld
188	Eligius Vanwechel	Eligius.Vanwechel@secretresearch.tld
82	Tramaine Scruton	Tramaine.Scruton@secretresearch.tld
1257	Alwin Roewe	Alwin.Roewe@secretresearch.tld
635	Tekla Gallerani	Tekla.Gallerani@secretresearch.tld
70	Argos Cumings	Argos.Cumings@secretresearch.tld
1310	Barsha Murrain	Barsha.Murrain@secretresearch.tld
1328	Xiomara Plowe	Xiomara.Plowe@secretresearch.tld
1367	Latorray Delawder	Latorray.Delawder@secretresearch.tld
698	Iha Em	Iha.Em@secretresearch.tld
462	Angela Hager	Angela.Hager@secretresearch.tld
252	Datja Carner	Datja.Carner@secretresearch.tld
629	Lotta Francoise	Lotta.Francoise@secretresearch.tld
1344	Evi Heidebrecht	Evi.Heidebrecht@secretresearch.tld
820	Yae Timblin	Yae.Timblin@secretresearch.tld
1315	Mamie Howen	Mamie.Howen@secretresearch.tld
783	Safiyya Letchaw	Safiyya.Letchaw@secretresearch.tld
523	Patrizio Iliffe	Patrizio.Iliffe@secretresearch.tld
518	Krosa Kuney	Krosa.Kuney@secretresearch.tld
693	Pilib Seu	Pilib.Seu@secretresearch.tld
641	Latoya Lovato	Latoya.Lovato@secretresearch.tld
603	Hannele Kneifl	Hannele.Kneifl@secretresearch.tld
1393	Demetria Tramonti	Demetria.Tramonti@secretresearch.tld
196	Shoko Moel	Shoko.Moel@secretresearch.tld
67	Shahaama Berum	Shahaama.Berum@secretresearch.tld
711	Iravati Pizzolato	Iravati.Pizzolato@secretresearch.tld
459	Riddhi Colesar	Riddhi.Colesar@secretresearch.tld
530	Caryn Grudzinski	Caryn.Grudzinski@secretresearch.tld
1742	Taliyah Jafari	Taliyah.Jafari@secretresearch.tld
1174	Sigurd Mahli	Sigurd.Mahli@secretresearch.tld
1722	Tera Vigen	Tera.Vigen@secretresearch.tld
1020	Barbie Dubonnet	Barbie.Dubonnet@secretresearch.tld
736	Loris Elder	Loris.Elder@secretresearch.tld
1739	Haimo Huesso	Haimo.Huesso@secretresearch.tld
1661	Liska Lambourne	Liska.Lambourne@secretresearch.tld
1045	Flaco Reyome	Flaco.Reyome@secretresearch.tld
524	Maeko Kontogiannis	Maeko.Kontogiannis@secretresearch.tld
1786	Ernst Bengtson	Ernst.Bengtson@secretresearch.tld
235	Dorel Brandt	Dorel.Brandt@secretresearch.tld
1508	Gerakna Ryneer	Gerakna.Ryneer@secretresearch.tld
467	Marek Mancino	Marek.Mancino@secretresearch.tld
1147	Este Manon	Este.Manon@secretresearch.tld
185	Madaleno Bercier	Madaleno.Bercier@secretresearch.tld
1606	Mya Swilley	Mya.Swilley@secretresearch.tld
1915	Korechika Mamer	Korechika.Mamer@secretresearch.tld
767	Inejiro Javens	Inejiro.Javens@secretresearch.tld
1701	Shizu Capener	Shizu.Capener@secretresearch.tld
939	Frangiskos Truly	Frangiskos.Truly@secretresearch.tld
678	Jace Ruotolo	Jace.Ruotolo@secretresearch.tld
27	Igone Rader	Igone.Rader@secretresearch.tld
1489	Chikafusa Koolman	Chikafusa.Koolman@secretresearch.tld
343	Cherry Dolence	Cherry.Dolence@secretresearch.tld
1998	Brys Robak	Brys.Robak@secretresearch.tld
225	Sanaaa Yoneda	Sanaaa.Yoneda@secretresearch.tld
1681	Aida Appelt	Aida.Appelt@secretresearch.tld
1420	Brenda Varley	Brenda.Varley@secretresearch.tld
589	Sam Mattheis	Sam.Mattheis@secretresearch.tld
1488	Galyn Stobie	Galyn.Stobie@secretresearch.tld
1810	Galni Mclarty	Galni.Mclarty@secretresearch.tld
1249	Dayu Vient	Dayu.Vient@secretresearch.tld
64	Miles Manero	Miles.Manero@secretresearch.tld
69	Heidelore Ruschmann	Heidelore.Ruschmann@secretresearch.tld
1520	Roseann Ora	Roseann.Ora@secretresearch.tld
1697	Dewayne Buttermore	Dewayne.Buttermore@secretresearch.tld
1547	Jonam Steinle	Jonam.Steinle@secretresearch.tld
540	Jamaall Rohen	Jamaall.Rohen@secretresearch.tld
1235	Latrice Ghrist	Latrice.Ghrist@secretresearch.tld
378	Akila Terlizzi	Akila.Terlizzi@secretresearch.tld
497	Oleka Tommasino	Oleka.Tommasino@secretresearch.tld
1959	Solymar Palisbo	Solymar.Palisbo@secretresearch.tld
145	Taaliba Debrita	Taaliba.Debrita@secretresearch.tld
1416	Edmonda Ditullio	Edmonda.Ditullio@secretresearch.tld
526	Kinaana Kodera	Kinaana.Kodera@secretresearch.tld
441	Waatiq Perico	Waatiq.Perico@secretresearch.tld
332	Coch Foulger	Coch.Foulger@secretresearch.tld
1901	Mitsuo Neff	Mitsuo.Neff@secretresearch.tld
1715	Martino Nommay	Martino.Nommay@secretresearch.tld
1395	Razan Symons	Razan.Symons@secretresearch.tld
1422	Lisandro Harpst	Lisandro.Harpst@secretresearch.tld
1406	Kerry Navarrate	Kerry.Navarrate@secretresearch.tld
537	Zumruda Fishman	Zumruda.Fishman@secretresearch.tld
1238	Ilsegret Akerley	Ilsegret.Akerley@secretresearch.tld
784	Matteo Mestre	Matteo.Mestre@secretresearch.tld
805	Takuboku Aicklen	Takuboku.Aicklen@secretresearch.tld
331	Analeigh Sebring	Analeigh.Sebring@secretresearch.tld
928	Dmitreeva Ellsbury	Dmitreeva.Ellsbury@secretresearch.tld
286	Gayle Wingrove	Gayle.Wingrove@secretresearch.tld
2006	Emese Abadie	Emese.Abadie@secretresearch.tld
281	Aleece Thorn	Aleece.Thorn@secretresearch.tld
1046	Shontaya Klitzner	Shontaya.Klitzner@secretresearch.tld
2061	Jascha Plantier	Jascha.Plantier@secretresearch.tld
1157	Sozui Jalovel	Sozui.Jalovel@secretresearch.tld
509	Prebrana Kekiwi	Prebrana.Kekiwi@secretresearch.tld
1840	Lonce Jiron	Lonce.Jiron@secretresearch.tld
1361	Gwalchmei Halbert	Gwalchmei.Halbert@secretresearch.tld
1895	Tahlia Francisque	Tahlia.Francisque@secretresearch.tld
344	Maxime Higgs	Maxime.Higgs@secretresearch.tld
982	Castalia Ryser	Castalia.Ryser@secretresearch.tld
1514	Ksniatintsa Fuchser	Ksniatintsa.Fuchser@secretresearch.tld
122	Jeanine Garoner	Jeanine.Garoner@secretresearch.tld
1775	Salamon Julien	Salamon.Julien@secretresearch.tld
1846	YolandaAbigail Insogna	YolandaAbigail.Insogna@secretresearch.tld
992	Phaethon Gearon	Phaethon.Gearon@secretresearch.tld
2217	Avigdor Puglia	Avigdor.Puglia@secretresearch.tld
814	Lecia Baumeister	Lecia.Baumeister@secretresearch.tld
507	Leonel Meche	Leonel.Meche@secretresearch.tld
1696	Okichi Spradlin	Okichi.Spradlin@secretresearch.tld
1878	Rufio Guitard	Rufio.Guitard@secretresearch.tld
362	Ramonda Tripplett	Ramonda.Tripplett@secretresearch.tld
377	Prokop Limage	Prokop.Limage@secretresearch.tld
1106	Araceli Ertel	Araceli.Ertel@secretresearch.tld
551	Aeacus Veloz	Aeacus.Veloz@secretresearch.tld
1919	Ceanna Vangoff	Ceanna.Vangoff@secretresearch.tld
1378	Nazarova Abundis	Nazarova.Abundis@secretresearch.tld
48	Adelio Zahner	Adelio.Zahner@secretresearch.tld
627	Blossom Quero	Blossom.Quero@secretresearch.tld
1072	Honami Docherty	Honami.Docherty@secretresearch.tld
1305	Bibiane Eddens	Bibiane.Eddens@secretresearch.tld
94	Casandra Shire	Casandra.Shire@secretresearch.tld
1607	Vilena Orillion	Vilena.Orillion@secretresearch.tld
1221	Tanja Lockmiller	Tanja.Lockmiller@secretresearch.tld
1277	AbdulHaadi Castellan	AbdulHaadi.Castellan@secretresearch.tld
1745	Fokina Poythress	Fokina.Poythress@secretresearch.tld
1436	Ulia Kaines	Ulia.Kaines@secretresearch.tld
815	Fanaila Torain	Fanaila.Torain@secretresearch.tld
730	Jermaine Mandahl	Jermaine.Mandahl@secretresearch.tld
508	Jake Bober	Jake.Bober@secretresearch.tld
1851	Fovro Chung	Fovro.Chung@secretresearch.tld
1271	Rosamar Wish	Rosamar.Wish@secretresearch.tld
1469	Iken Siluis	Iken.Siluis@secretresearch.tld
146	Liselotte Groleau	Liselotte.Groleau@secretresearch.tld
2170	Marce Fiscalini	Marce.Fiscalini@secretresearch.tld
1591	Bennet Melendes	Bennet.Melendes@secretresearch.tld
1478	Herberta Fullenwider	Herberta.Fullenwider@secretresearch.tld
2053	Torokanova Kosloski	Torokanova.Kosloski@secretresearch.tld
404	Pazia Comnick	Pazia.Comnick@secretresearch.tld
1792	Stacie Ciesielski	Stacie.Ciesielski@secretresearch.tld
910	Benita Hatzenbuehler	Benita.Hatzenbuehler@secretresearch.tld
1025	Shaka Enget	Shaka.Enget@secretresearch.tld
1890	Majdiyya Bonsall	Majdiyya.Bonsall@secretresearch.tld
2097	Gerharde Heidtbrink	Gerharde.Heidtbrink@secretresearch.tld
1247	Honey Eidinger	Honey.Eidinger@secretresearch.tld
1103	Adamanta Schlitt	Adamanta.Schlitt@secretresearch.tld
2029	Faqueza Correll	Faqueza.Correll@secretresearch.tld
1506	Mirra Tygart	Mirra.Tygart@secretresearch.tld
1773	Cloelia Shein	Cloelia.Shein@secretresearch.tld
292	Rosalinde Leinbach	Rosalinde.Leinbach@secretresearch.tld
550	Detana Trautz	Detana.Trautz@secretresearch.tld
287	Favio Mutherspaw	Favio.Mutherspaw@secretresearch.tld
2026	Janny Mccaine	Janny.Mccaine@secretresearch.tld
670	Toyoaki Richoux	Toyoaki.Richoux@secretresearch.tld
1582	Clementine Barga	Clementine.Barga@secretresearch.tld
1289	Volotkoveia Klees	Volotkoveia.Klees@secretresearch.tld
338	Marisa Roufs	Marisa.Roufs@secretresearch.tld
544	Lacrasha Slavick	Lacrasha.Slavick@secretresearch.tld
1109	Cadell Parral	Cadell.Parral@secretresearch.tld
1578	Angel Delcarmen	Angel.Delcarmen@secretresearch.tld
746	Arno Barquera	Arno.Barquera@secretresearch.tld
997	Bussho Oda	Bussho.Oda@secretresearch.tld
1855	Baucis Campese	Baucis.Campese@secretresearch.tld
330	Ieshia Demarrais	Ieshia.Demarrais@secretresearch.tld
1634	Ehud Bazzle	Ehud.Bazzle@secretresearch.tld
817	Jamie Berrie	Jamie.Berrie@secretresearch.tld
757	Emily Mileski	Emily.Mileski@secretresearch.tld
1177	Leonarda Imai	Leonarda.Imai@secretresearch.tld
236	Bruna Nervis	Bruna.Nervis@secretresearch.tld
221	Sebastiano Lockyer	Sebastiano.Lockyer@secretresearch.tld
1074	Yukako Klingbeil	Yukako.Klingbeil@secretresearch.tld
1102	Barbi Denery	Barbi.Denery@secretresearch.tld
822	Feli Trainham	Feli.Trainham@secretresearch.tld
1401	Hagibis Kihn	Hagibis.Kihn@secretresearch.tld
1407	Cady Schweizer	Cady.Schweizer@secretresearch.tld
339	Javier Hausauer	Javier.Hausauer@secretresearch.tld
846	Zeanes Hinsley	Zeanes.Hinsley@secretresearch.tld
861	ra Jansons	ra.Jansons@secretresearch.tld
54	Amalia Krajcik	Amalia.Krajcik@secretresearch.tld
1405	Kyledyr Constantino	Kyledyr.Constantino@secretresearch.tld
370	Sajag Traunfeld	Sajag.Traunfeld@secretresearch.tld
2071	Janeka Hochhalter	Janeka.Hochhalter@secretresearch.tld
826	Aurelian Fealy	Aurelian.Fealy@secretresearch.tld
985	Vasyklo Netz	Vasyklo.Netz@secretresearch.tld
226	Kiyoemon Wesby	Kiyoemon.Wesby@secretresearch.tld
2066	Aedd Lurye	Aedd.Lurye@secretresearch.tld
722	Sherry Mullennix	Sherry.Mullennix@secretresearch.tld
1095	Gerlind Boomershine	Gerlind.Boomershine@secretresearch.tld
290	Aberthol Wilczak	Aberthol.Wilczak@secretresearch.tld
148	Toshimichi Speth	Toshimichi.Speth@secretresearch.tld
1979	Alka Groshek	Alka.Groshek@secretresearch.tld
307	Tango Constable	Tango.Constable@secretresearch.tld
1830	Shoichi Mathey	Shoichi.Mathey@secretresearch.tld
1350	Matashichi Butman	Matashichi.Butman@secretresearch.tld
671	Rico Boronat	Rico.Boronat@secretresearch.tld
1419	Maela Chamley	Maela.Chamley@secretresearch.tld
1755	Saer Dillenbeck	Saer.Dillenbeck@secretresearch.tld
800	Claudios Lemmert	Claudios.Lemmert@secretresearch.tld
1280	Luca Dolbin	Luca.Dolbin@secretresearch.tld
1467	Walto Messman	Walto.Messman@secretresearch.tld
762	Tamiko Docken	Tamiko.Docken@secretresearch.tld
851	Marna Crookshanks	Marna.Crookshanks@secretresearch.tld
198	Julien Triveno	Julien.Triveno@secretresearch.tld
1439	Amir Conkling	Amir.Conkling@secretresearch.tld
513	Christelle Garced	Christelle.Garced@secretresearch.tld
1624	Danaus Dacanay	Danaus.Dacanay@secretresearch.tld
1145	Gisselle Teigen	Gisselle.Teigen@secretresearch.tld
1479	Primavera Huffstickler	Primavera.Huffstickler@secretresearch.tld
2022	Donald Ludlow	Donald.Ludlow@secretresearch.tld
363	Momus Rivadeneira	Momus.Rivadeneira@secretresearch.tld
1564	Jada Grech	Jada.Grech@secretresearch.tld
838	Adolfa Leuthauser	Adolfa.Leuthauser@secretresearch.tld
2151	Gigi Barfuss	Gigi.Barfuss@secretresearch.tld
1852	Eber Ebinger	Eber.Ebinger@secretresearch.tld
93	Clmentine Engellant	Clmentine.Engellant@secretresearch.tld
2173	Cellina Schlosser	Cellina.Schlosser@secretresearch.tld
798	Vrotsislava Bouman	Vrotsislava.Bouman@secretresearch.tld
158	Kea Croshaw	Kea.Croshaw@secretresearch.tld
2115	Xylina Mcnicoll	Xylina.Mcnicoll@secretresearch.tld
97	Urako Staker	Urako.Staker@secretresearch.tld
2069	Guiseppa Thurstonson	Guiseppa.Thurstonson@secretresearch.tld
306	Shijo Mccullon	Shijo.Mccullon@secretresearch.tld
1254	Abbie Manjarres	Abbie.Manjarres@secretresearch.tld
1762	Jafar Abbenante	Jafar.Abbenante@secretresearch.tld
506	Tasia Ellenburg	Tasia.Ellenburg@secretresearch.tld
172	Estrelle Wedgworth	Estrelle.Wedgworth@secretresearch.tld
383	Lashaune Lava	Lashaune.Lava@secretresearch.tld
1017	Raffaello Herard	Raffaello.Herard@secretresearch.tld
553	Bernharda Riese	Bernharda.Riese@secretresearch.tld
944	Parma Borgese	Parma.Borgese@secretresearch.tld
543	Inayat Deignan	Inayat.Deignan@secretresearch.tld
853	Burkhard Hussein	Burkhard.Hussein@secretresearch.tld
1431	Uliaanitsa Samul	Uliaanitsa.Samul@secretresearch.tld
1454	Faatin Bonn	Faatin.Bonn@secretresearch.tld
1115	Jokin Hytros	Jokin.Hytros@secretresearch.tld
1356	Apela Brewton	Apela.Brewton@secretresearch.tld
277	Hellmut Tousant	Hellmut.Tousant@secretresearch.tld
365	Kerstine Finders	Kerstine.Finders@secretresearch.tld
959	NewyddIlyn Borchers	NewyddIlyn.Borchers@secretresearch.tld
651	Elvis Rocasah	Elvis.Rocasah@secretresearch.tld
1460	Gengyo Cumbass	Gengyo.Cumbass@secretresearch.tld
1133	Vychan Stotz	Vychan.Stotz@secretresearch.tld
871	Ora Poleyestewa	Ora.Poleyestewa@secretresearch.tld
1071	Hoitsu Mccahill	Hoitsu.Mccahill@secretresearch.tld
1076	Ariana Berardinelli	Ariana.Berardinelli@secretresearch.tld
180	Natascha Neverson	Natascha.Neverson@secretresearch.tld
1144	Pete Salada	Pete.Salada@secretresearch.tld
1248	Korotsek Baltzell	Korotsek.Baltzell@secretresearch.tld
170	Swidbert Repka	Swidbert.Repka@secretresearch.tld
1118	Clarine Waynick	Clarine.Waynick@secretresearch.tld
101	Hilmiyya Pagon	Hilmiyya.Pagon@secretresearch.tld
480	Isadoro Bently	Isadoro.Bently@secretresearch.tld
1185	Benignus Diec	Benignus.Diec@secretresearch.tld
1613	Tetsui Zekria	Tetsui.Zekria@secretresearch.tld
1302	Dunja Luchenbill	Dunja.Luchenbill@secretresearch.tld
1433	Kazuhiko Campas	Kazuhiko.Campas@secretresearch.tld
1798	Amba Guidos	Amba.Guidos@secretresearch.tld
697	Kywrkh Tomaino	Kywrkh.Tomaino@secretresearch.tld
1464	Saahir Muto	Saahir.Muto@secretresearch.tld
1056	Lowell Kostka	Lowell.Kostka@secretresearch.tld
1468	Korotskovaia Stowell	Korotskovaia.Stowell@secretresearch.tld
1623	Ivonne Elreda	Ivonne.Elreda@secretresearch.tld
809	Faas Chitty	Faas.Chitty@secretresearch.tld
1341	Ryo Zanders	Ryo.Zanders@secretresearch.tld
1101	Lubachitsa Husch	Lubachitsa.Husch@secretresearch.tld
1952	Cato Himmelwright	Cato.Himmelwright@secretresearch.tld
1942	Yayang Kuczenski	Yayang.Kuczenski@secretresearch.tld
1011	Mita Schwering	Mita.Schwering@secretresearch.tld
243	Juichi Porep	Juichi.Porep@secretresearch.tld
1698	Phoolan Mackie	Phoolan.Mackie@secretresearch.tld
53	Ashley Kesselring	Ashley.Kesselring@secretresearch.tld
529	Virgilio Englander	Virgilio.Englander@secretresearch.tld
1660	Sharisha Gebhart	Sharisha.Gebhart@secretresearch.tld
1512	Otylia Lowdermilk	Otylia.Lowdermilk@secretresearch.tld
17	Alon Paulik	Alon.Paulik@secretresearch.tld
1693	Asisa Finke	Asisa.Finke@secretresearch.tld
1438	Adwr Liiv	Adwr.Liiv@secretresearch.tld
1859	Seinosuke Grimaldi	Seinosuke.Grimaldi@secretresearch.tld
9	Sadayoshi Englemann	Sadayoshi.Englemann@secretresearch.tld
1130	Teodoro Coyt	Teodoro.Coyt@secretresearch.tld
1073	Hilario Kovats	Hilario.Kovats@secretresearch.tld
505	Thanaa Bustos	Thanaa.Bustos@secretresearch.tld
987	Ladinka Ehiginator	Ladinka.Ehiginator@secretresearch.tld
1682	LaReina Iqbal	LaReina.Iqbal@secretresearch.tld
587	Timotheos Cazaree	Timotheos.Cazaree@secretresearch.tld
1541	Zotico Cerrano	Zotico.Cerrano@secretresearch.tld
663	Hesperos Schindele	Hesperos.Schindele@secretresearch.tld
1294	Terao Asley	Terao.Asley@secretresearch.tld
700	Eijiro Majied	Eijiro.Majied@secretresearch.tld
1769	Sebina Prakoth	Sebina.Prakoth@secretresearch.tld
845	Ludovica Diekmann	Ludovica.Diekmann@secretresearch.tld
879	Yuka Heikes	Yuka.Heikes@secretresearch.tld
1638	Sanya Parma	Sanya.Parma@secretresearch.tld
328	Zytka Skeete	Zytka.Skeete@secretresearch.tld
742	Wilhelm Juart	Wilhelm.Juart@secretresearch.tld
364	Lamesha Neher	Lamesha.Neher@secretresearch.tld
696	Triesha Carland	Triesha.Carland@secretresearch.tld
425	Radosha Ferioli	Radosha.Ferioli@secretresearch.tld
168	Twain Sannon	Twain.Sannon@secretresearch.tld
1549	Jennie Schatzel	Jennie.Schatzel@secretresearch.tld
1335	Gyoergy Rakoci	Gyoergy.Rakoci@secretresearch.tld
5	Aishwarya Hallmon	Aishwarya.Hallmon@secretresearch.tld
980	Kotaro Leitheiser	Kotaro.Leitheiser@secretresearch.tld
752	Garrett Blair	Garrett.Blair@secretresearch.tld
840	Gerke Ostermeyer	Gerke.Ostermeyer@secretresearch.tld
1212	Kozue Efthimiou	Kozue.Efthimiou@secretresearch.tld
594	Tamah Misemer	Tamah.Misemer@secretresearch.tld
1860	Florentyna Cost	Florentyna.Cost@secretresearch.tld
702	Newa Sabina	Newa.Sabina@secretresearch.tld
532	Ashakiran Bingley	Ashakiran.Bingley@secretresearch.tld
657	Krystyn Slutsky	Krystyn.Slutsky@secretresearch.tld
34	Elan Malloch	Elan.Malloch@secretresearch.tld
729	Materia Newsum	Materia.Newsum@secretresearch.tld
868	Delmara Mrazek	Delmara.Mrazek@secretresearch.tld
835	Tekusa Houlberg	Tekusa.Houlberg@secretresearch.tld
501	Maksimina Pecinousky	Maksimina.Pecinousky@secretresearch.tld
866	Oscar Formica	Oscar.Formica@secretresearch.tld
836	Luithilde Brandolino	Luithilde.Brandolino@secretresearch.tld
1872	Matro Newell	Matro.Newell@secretresearch.tld
1567	Gerik Ferries	Gerik.Ferries@secretresearch.tld
1340	Iselle Churches	Iselle.Churches@secretresearch.tld
1164	Muzammil Mutter	Muzammil.Mutter@secretresearch.tld
935	Bernhareda Mcclenny	Bernhareda.Mcclenny@secretresearch.tld
909	Emmanoul Gianunzio	Emmanoul.Gianunzio@secretresearch.tld
674	Angelica Maccarter	Angelica.Maccarter@secretresearch.tld
385	Paco Filary	Paco.Filary@secretresearch.tld
1823	Jamuna Orts	Jamuna.Orts@secretresearch.tld
953	Pascale Skripko	Pascale.Skripko@secretresearch.tld
106	Pantyslawa Pemble	Pantyslawa.Pemble@secretresearch.tld
1149	Abderus Boston	Abderus.Boston@secretresearch.tld
1958	Alala Cilento	Alala.Cilento@secretresearch.tld
56	Izazkun Suleski	Izazkun.Suleski@secretresearch.tld
\.



COPY "Preference" ("ID") FROM stdin;
\.



COPY "ProgramCommittee" ("ID", label) FROM stdin;
668	SIGXYZ 2014 Program Committee
1005	XSWC 2015 Program Committee
565	XSWC 2014 Program Committee
1266	YSWC 2015 Program Committee
1330	ZSWC 2015 Program Committee
1267	YSWC 2014 Program Committee
318	ZSWC 2014 Program Committee
\.



COPY "ProgramCommitteeChair" ("ID") FROM stdin;
24
200
473
535
678
1034
\.



COPY "ProgramCommitteeMember" ("ID", "maxPapers", "addedBy") FROM stdin;
316	\N	\N
666	\N	\N
200	\N	\N
247	\N	\N
635	\N	\N
1328	\N	\N
1367	\N	\N
45	\N	\N
241	\N	\N
88	\N	\N
273	\N	\N
523	\N	\N
459	\N	\N
84	\N	\N
1297	\N	\N
644	\N	\N
1701	\N	\N
678	\N	\N
743	\N	\N
1810	\N	\N
31	\N	\N
228	\N	\N
219	\N	\N
541	\N	\N
1034	\N	\N
535	\N	\N
411	\N	\N
487	\N	\N
473	\N	\N
637	\N	\N
393	\N	\N
1089	\N	\N
24	\N	\N
\.



COPY "Rejection" ("ID") FROM stdin;
\.



COPY "Review" ("ID", "writtenBy", comment) FROM stdin;
2	3	The paper "XSWC-2014" is mainly about XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
4	5	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
8	9	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
15	16	The paper "XSWC-2014" is mainly about XSWC-2014.\nThough apparently a contribution of remarkable novelty, 'XSWC-2014' also has its weaknesses.
21	22	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
23	24	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
26	27	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
55	56	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
63	64	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
73	74	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
75	76	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
77	78	The paper "XSWC-2014" is mainly about XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
79	80	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
87	88	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
95	96	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
100	101	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
102	103	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
144	145	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
156	157	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
166	70	The paper "XSWC-2014" is mainly about XSWC-2014.\nThough apparently a contribution of remarkable novelty, 'XSWC-2014' also has its weaknesses.
18	17	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
167	168	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
169	170	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
182	183	The paper "XSWC-2014" is mainly about XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
202	203	The paper "XSWC-2014" is mainly about XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
204	16	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
213	214	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
215	216	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
220	221	The paper "XSWC-2014" is mainly about XSWC-2014.\nThough apparently a contribution of remarkable novelty, 'XSWC-2014' also has its weaknesses.
227	228	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
229	230	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
253	254	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
261	91	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
272	273	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
280	276	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
297	298	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
305	306	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
311	312	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
314	315	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
323	324	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
325	326	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
336	235	The paper "XSWC-2014" is mainly about XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
342	343	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
366	367	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
368	369	The paper "XSWC-2014" is mainly about XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
382	383	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
142	138	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
384	285	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
389	390	The paper "XSWC-2014" is mainly about XSWC-2014.\nThough apparently a contribution of remarkable novelty, 'XSWC-2014' also has its weaknesses.
406	163	The paper "XSWC-2014" is mainly about XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
413	414	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
424	425	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
437	326	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
466	467	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
468	469	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
470	225	The paper "YSWC-2015" is mainly about YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
472	473	The paper "YSWC-2014" is mainly about YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
478	479	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
490	304	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
496	497	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
499	500	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
515	516	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
517	518	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
527	528	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
534	535	The paper "XSWC-2014" is mainly about XSWC-2014.\nThough apparently a contribution of remarkable novelty, 'XSWC-2014' also has its weaknesses.
542	543	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
546	196	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
547	181	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
0	1	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
598	599	The paper "YSWC-2014" is mainly about YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
452	448	The paper "YSWC-2014" is mainly about YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
1408	795	The paper "XSWC-2015" is mainly about XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
125	47	The paper "XSWC-2015" is mainly about XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
989	941	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
1997	1998	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
1186	1185	The paper "YSWC-2014" is mainly about YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
131	128	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
135	128	The paper "XSWC-2014" is mainly about XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
266	262	The paper "XSWC-2014" is mainly about XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
731	230	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
770	769	The paper "YSWC-2015" is mainly about YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
1756	1174	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
554	555	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
434	426	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
576	505	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
578	417	The paper "XSWC-2014" is mainly about XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
586	587	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
600	601	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
602	603	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
605	161	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
612	613	The paper "XSWC-2014" is mainly about XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
619	71	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
650	460	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
654	640	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
655	530	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
660	393	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
675	676	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
677	678	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
679	625	The paper "YSWC-2014" is mainly about YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
683	684	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
692	693	The paper "XSWC-2014" is mainly about XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
699	700	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
701	702	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
710	711	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
715	716	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
717	718	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
484	483	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
719	411	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
720	721	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
735	736	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
738	739	The paper "XSWC-2014" is mainly about XSWC-2014.\nThough apparently a contribution of remarkable novelty, 'XSWC-2014' also has its weaknesses.
747	34	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
748	697	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
749	750	The paper "XSWC-2014" is mainly about XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
760	462	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
561	539	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
764	378	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
782	783	The paper "YSWC-2014" is mainly about YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
792	793	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
799	638	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
758	757	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
807	276	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
808	809	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
816	817	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
819	820	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
823	393	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
834	643	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
839	840	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
841	275	The paper "YSWC-2014" is mainly about YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
206	205	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
818	742	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
843	1	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
844	845	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
860	162	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
870	871	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
873	836	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
726	725	The paper "YSWC-2014" is mainly about YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
878	529	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
882	611	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
898	419	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
899	709	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
900	641	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
917	188	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
931	684	The paper "YSWC-2014" is mainly about YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
932	933	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
934	935	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
938	939	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
951	487	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
952	953	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
958	959	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
960	635	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
962	927	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
963	868	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
970	3	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
971	766	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
972	973	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
974	975	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
984	535	The paper "XSWC-2015" is mainly about XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
491	445	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
996	997	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
998	750	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1002	1003	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
1007	723	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1026	82	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1027	275	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1035	76	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1039	645	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1040	1041	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
1043	674	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
1044	1045	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
667	666	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1053	1054	The paper "YSWC-2014" is mainly about YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
1055	1056	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1075	812	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
858	200	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1079	1080	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
1081	248	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1263	1262	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1409	766	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1410	626	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1414	109	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1415	1416	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
1417	447	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
41	40	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThough apparently a contribution of remarkable novelty, 'XSWC-2014' also has its weaknesses.
1421	626	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
1430	1431	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
1432	995	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1441	1442	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
1022	409	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1447	1093	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1457	1052	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1459	1460	The paper "XSWC-2015" is mainly about XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
1461	74	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThough apparently a contribution of remarkable novelty, 'XSWC-2014' also has its weaknesses.
1462	84	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
1470	1393	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1475	214	The paper "XSWC-2015" is mainly about XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
428	426	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1485	1249	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1486	83	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1487	663	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
1491	1492	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
1493	1494	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1496	1008	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1498	1488	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1501	644	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1502	69	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1107	330	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1504	1190	The paper "XSWC-2015" is mainly about XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
1028	85	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1511	1512	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
890	607	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
585	500	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
1519	1520	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
920	50	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
293	279	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
455	448	The paper "YSWC-2015" is mainly about YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
664	532	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1536	199	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
570	568	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
574	573	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1546	1547	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1551	56	The paper "XSWC-2015" is mainly about XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
1556	632	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1560	687	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
580	579	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
1228	53	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThough apparently a contribution of remarkable novelty, 'XSWC-2014' also has its weaknesses.
1566	1567	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
560	539	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
1572	1360	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1471	629	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1573	549	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
1580	76	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1589	473	The paper "YSWC-2015" is mainly about YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
1592	767	The paper "YSWC-2015" is mainly about YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
1593	1420	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1594	1199	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1597	425	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThough apparently a contribution of remarkable novelty, 'XSWC-2014' also has its weaknesses.
1601	442	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1383	234	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1612	633	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1615	724	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
948	1	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1628	1175	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
1336	1335	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
1632	1633	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
495	445	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1644	1488	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
622	621	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1649	673	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1655	288	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1155	1154	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1024	1023	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
1264	635	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
1657	71	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1550	1549	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1665	640	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1672	1508	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1673	1674	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1676	625	The paper "YSWC-2015" is mainly about YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
1677	237	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
875	874	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1679	636	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1685	766	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1686	1687	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1625	641	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1700	1701	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
930	929	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
1037	590	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1643	1003	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
1131	1130	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1713	642	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
173	172	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1716	242	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1720	957	The paper "YSWC-2015" is mainly about YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
1721	1722	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1723	599	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1724	1725	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
1734	69	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1736	638	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1718	687	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
1683	1682	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
1744	221	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1726	459	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
1746	393	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1569	1438	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1750	919	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
1614	1613	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1754	1424	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1427	45	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1768	1769	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
1603	908	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1774	33	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
1223	188	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1214	1212	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1776	1169	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
1778	1697	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1465	1464	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1779	767	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1353	1351	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
780	777	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
1444	241	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
2065	1175	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
848	46	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1523	1315	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1791	1120	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1794	942	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
317	316	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
753	752	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1802	223	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1805	240	The paper "XSWC-2014" is mainly about XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1809	1810	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
268	262	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
1253	82	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1814	233	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1816	162	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
1170	244	The paper "XSWC-2014" is mainly about XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1731	203	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1818	1578	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
1821	661	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
886	236	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1273	275	The paper "YSWC-2015" is mainly about YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
1828	1674	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1831	673	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1836	106	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1837	1071	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
1849	473	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
859	200	The paper "XSWC-2014" is mainly about XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1641	245	The paper "XSWC-2014" is mainly about XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1807	1739	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1150	1149	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1854	31	The paper "XSWC-2014" is mainly about XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1858	83	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1412	684	The paper "YSWC-2015" is mainly about YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
1864	12	The paper "XSWC-2014" is mainly about XSWC-2014.\nThough apparently a contribution of remarkable novelty, 'XSWC-2014' also has its weaknesses.
1866	1231	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
1867	1380	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1869	1093	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1874	1248	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1875	1681	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
1265	635	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1752	1017	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1765	1764	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1702	67	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
1088	1089	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1104	31	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1114	1115	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1119	1120	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1121	739	The paper "XSWC-2015" is mainly about XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
58	57	The paper "YSWC-2014" is mainly about YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
1146	1147	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1152	927	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
977	254	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1163	1164	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1166	1167	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1168	1169	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThough apparently a contribution of remarkable novelty, 'XSWC-2014' also has its weaknesses.
1176	1177	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
1178	369	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1201	390	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1203	1118	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1204	1205	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1206	420	The paper "XSWC-2015" is mainly about XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
945	944	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1208	686	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1209	241	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1220	739	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1227	813	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1108	92	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
1233	78	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1234	634	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1246	1118	The paper "XSWC-2015" is mainly about XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
1269	1270	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1275	611	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
1278	1279	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
1282	242	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1284	70	The paper "XSWC-2015" is mainly about XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
1286	238	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1287	825	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1288	993	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1162	183	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1296	1297	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
1300	232	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1301	631	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1306	1008	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1309	221	The paper "XSWC-2015" is mainly about XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
1313	45	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1316	850	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1318	302	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1319	84	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1320	953	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
431	426	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1338	673	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1339	1340	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1345	980	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1347	234	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
520	197	The paper "XSWC-2015" is mainly about XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
1251	82	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
990	941	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThough apparently a contribution of remarkable novelty, 'XSWC-2014' also has its weaknesses.
1373	246	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1374	78	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
649	647	The paper "XSWC-2015" is mainly about XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
1375	1376	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1379	1380	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
121	116	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1382	163	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1386	1387	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThough apparently a contribution of remarkable novelty, 'XSWC-2014' also has its weaknesses.
1389	367	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
271	262	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1348	1132	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
1394	475	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1960	632	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1329	1328	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1916	1915	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
863	107	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
155	151	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
2004	24	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1856	1855	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1928	644	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
2014	1437	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
1885	442	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
451	448	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1031	85	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
2019	525	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1428	45	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
1804	444	The paper "YSWC-2014" is mainly about YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
321	316	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
849	46	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
2028	1018	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1749	1748	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
2032	199	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
1545	909	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
2034	276	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
804	803	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
572	568	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1630	1629	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
2037	1468	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1113	1112	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
1719	687	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
399	16	The paper "XSWC-2015" is mainly about XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
691	690	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
2043	957	The paper "YSWC-2014" is mainly about YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
2045	1175	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1692	636	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
2047	225	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
1183	326	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
481	480	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
2046	225	The paper "YSWC-2014" is mainly about YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
2059	1905	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1084	418	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1873	1872	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
584	500	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
36	35	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
855	200	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
2068	164	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1096	417	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
2063	1488	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1732	203	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1012	1011	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
2075	1094	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
322	316	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
2081	1093	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1770	1020	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1898	1541	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1900	1555	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1826	788	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThough apparently a contribution of remarkable novelty, 'XSWC-2014' also has its weaknesses.
2087	240	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1834	524	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
2092	237	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
2093	1034	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
1990	1860	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
2094	390	The paper "XSWC-2015" is mainly about XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
2095	240	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1903	538	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1171	244	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
2100	1094	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
1216	1091	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
1812	189	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
2057	1019	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
2106	1123	The paper "XSWC-2015" is mainly about XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
1945	71	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
2042	9	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1258	1257	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1213	1212	The paper "XSWC-2015" is mainly about XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
2117	919	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
1908	185	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
2083	64	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThough apparently a contribution of remarkable novelty, 'XSWC-2014' also has its weaknesses.
1639	245	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
1097	417	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
880	879	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
2119	12	The paper "XSWC-2015" is mainly about XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
2126	637	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
2128	2129	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1365	613	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1760	987	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
2131	837	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
2132	414	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
2135	219	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
2138	239	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1311	1310	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1528	1481	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThough apparently a contribution of remarkable novelty, 'XSWC-2014' also has its weaknesses.
1727	459	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThough apparently a contribution of remarkable novelty, 'XSWC-2014' also has its weaknesses.
2142	526	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
1217	1091	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
2145	608	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1354	1351	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1784	795	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
2152	956	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1323	750	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1595	273	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThough apparently a contribution of remarkable novelty, 'XSWC-2014' also has its weaknesses.
1371	698	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
1838	708	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1581	34	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
2105	1623	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
680	180	The paper "XSWC-2015" is mainly about XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
902	201	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
595	594	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
2160	1231	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
2161	487	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
139	138	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1303	1302	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
779	777	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
2165	233	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1650	364	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1966	1637	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1913	1578	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThough apparently a contribution of remarkable novelty, 'XSWC-2014' also has its weaknesses.
1138	1137	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
2169	1073	The paper "XSWC-2015" is mainly about XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
1290	157	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
2000	1489	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
2089	162	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
1863	1297	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1694	1693	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
2125	540	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
2113	243	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
258	255	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1868	640	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1741	1295	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
2157	1942	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1142	1141	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
2176	80	The paper "XSWC-2015" is mainly about XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
1670	1393	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1456	1454	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThough apparently a contribution of remarkable novelty, 'XSWC-2014' also has its weaknesses.
358	352	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1402	462	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
2180	1638	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
2150	161	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
2185	22	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
1538	1537	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
2070	1423	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
2010	927	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1608	523	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
494	445	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
772	769	The paper "YSWC-2014" is mainly about YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
774	769	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
887	236	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
916	643	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1004	247	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1064	216	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1128	13	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1195	1194	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1241	1240	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1291	157	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1466	1387	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
1482	995	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1509	367	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
1530	631	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1598	709	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1640	245	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1689	1050	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1711	475	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1712	475	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
1787	993	The paper "XSWC-2015" is mainly about XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
1788	993	The paper "XSWC-2014" is mainly about XSWC-2014.\nThough apparently a contribution of remarkable novelty, 'XSWC-2014' also has its weaknesses.
1795	642	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1813	189	The paper "XSWC-2014" is mainly about XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1886	442	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
1983	1205	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1881	24	The paper "XSWC-2014" is mainly about XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1884	328	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
1887	541	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1870	1635	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1892	608	The paper "XSWC-2014" is mainly about XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1896	420	The paper "XSWC-2014" is mainly about XSWC-2014.\nThough apparently a contribution of remarkable novelty, 'XSWC-2014' also has its weaknesses.
1586	1052	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1110	1109	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
1906	956	The paper "XSWC-2014" is mainly about XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1153	277	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
905	201	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1917	1400	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1129	13	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1918	1054	The paper "YSWC-2015" is mainly about YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
1187	1185	The paper "YSWC-2015" is mainly about YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
1472	1344	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
210	209	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1920	1577	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1921	639	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1497	1144	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1922	1120	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1923	1660	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
1924	51	The paper "XSWC-2014" is mainly about XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1451	252	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1215	1212	The paper "XSWC-2014" is mainly about XSWC-2014.\nThough apparently a contribution of remarkable novelty, 'XSWC-2014' also has its weaknesses.
1691	1294	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1707	548	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1932	410	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
1933	813	The paper "XSWC-2014" is mainly about XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1936	1235	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1937	1938	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1939	232	The paper "XSWC-2014" is mainly about XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1940	22	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
1943	1073	The paper "XSWC-2014" is mainly about XSWC-2014.\nThough apparently a contribution of remarkable novelty, 'XSWC-2014' also has its weaknesses.
889	607	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1950	743	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1951	1952	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
1953	1167	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1954	33	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1957	1699	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1819	1661	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
1061	1060	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
1962	488	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1894	486	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1967	1698	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1970	1634	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
648	647	The paper "XSWC-2014" is mainly about XSWC-2014.\nThough apparently a contribution of remarkable novelty, 'XSWC-2014' also has its weaknesses.
1973	365	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
704	703	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
968	967	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
1006	247	The paper "XSWC-2014" is mainly about XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1976	626	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1969	1341	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1648	1199	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1931	767	The paper "YSWC-2014" is mainly about YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
1981	1205	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1842	1786	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1743	1742	The paper "ZSWC-2015" is mainly about ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1984	219	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
564	539	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1666	1101	The paper "XSWC-2014" is mainly about XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
1077	1076	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1989	589	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
1963	633	The paper "YSWC-2014" is mainly about YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1993	1167	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
1926	1356	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1996	1767	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1986	288	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
1987	288	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1991	27	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
2030	1671	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
2035	44	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
2079	223	In their paper 'YSWC-2015' the authors discuss YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
2080	223	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
2137	187	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
2190	1762	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1688	729	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
1902	538	The paper "XSWC-2015" is mainly about XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1964	633	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
904	201	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1548	634	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1255	1254	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
386	385	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1526	783	The paper "YSWC-2015" is mainly about YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
2021	506	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
2054	639	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1232	300	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
1912	1606	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1599	709	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1961	632	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1985	1080	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
1403	462	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
2206	391	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1434	1433	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1799	1798	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
1808	1115	The paper "XSWC-2015" is mainly about XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
1600	853	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1626	641	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
2007	645	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
2012	553	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
1947	419	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
2204	953	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
658	657	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
1513	88	In their paper 'XSWC-2014' the authors discuss XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
2064	1958	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
2101	33	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1825	788	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
1158	696	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
2146	1636	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
152	151	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
2219	487	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
2149	601	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
2085	1334	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
2154	1959	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
502	501	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
2226	1094	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
1368	1367	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1645	1199	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
2198	1543	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
2229	637	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1579	746	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1086	418	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
2052	1859	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
2188	219	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
2060	238	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
2213	608	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
2232	301	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
892	835	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
1455	1454	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
1085	418	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
357	352	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
2130	707	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
61	57	The paper "YSWC-2015" is mainly about YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
2136	187	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1668	1393	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1426	1425	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
2239	637	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1453	1452	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2014' also has its weaknesses.
2196	239	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
1824	1823	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThough apparently a contribution of remarkable novelty, 'ZSWC-2015' also has its weaknesses.
2171	663	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
2223	248	The paper "XSWC-2014" is mainly about XSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2014 in particular
2209	866	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1529	1481	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThough apparently a contribution of remarkable novelty, 'XSWC-2015' also has its weaknesses.
1033	85	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
1224	188	In their paper 'ZSWC-2014' the authors discuss ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
2245	813	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
2247	246	In their paper 'XSWC-2015' the authors discuss XSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight XSWC-2015 in particular
557	556	In their paper 'ZSWC-2015' the authors discuss ZSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2015 in particular
1803	444	The paper "YSWC-2015" is mainly about YSWC-2015.\nThough apparently a contribution of remarkable novelty, 'YSWC-2015' also has its weaknesses.
1780	163	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
2177	611	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2014 in particular
1134	1133	The paper "SIGXYZ-2014" is mainly about SIGXYZ-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight SIGXYZ-2014 in particular
2008	645	The paper "YSWC-2015" is mainly about YSWC-2015.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight YSWC-2015 in particular
2031	1671	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
652	651	The paper "ZSWC-2014" is mainly about ZSWC-2014.\nThis says about all about both its strong and weak suits.\nHowever, the reviewer would like to highlight ZSWC-2014 in particular
1759	987	In their paper 'YSWC-2014' the authors discuss YSWC-2014.\nThough apparently a contribution of remarkable novelty, 'YSWC-2014' also has its weaknesses.
616	615	In their paper 'SIGXYZ-2014' the authors discuss SIGXYZ-2014.\nThough apparently a contribution of remarkable novelty, 'SIGXYZ-2014' also has its weaknesses.
\.



COPY "Reviewer" ("ID") FROM stdin;
1
3
5
9
12
13
16
17
22
24
27
31
33
34
35
40
44
45
46
47
50
51
53
56
57
64
67
69
70
71
74
76
78
80
82
83
84
85
88
91
92
96
101
103
106
107
109
116
128
138
145
151
157
161
162
163
164
168
170
172
180
181
183
185
187
188
189
196
197
199
200
201
203
205
209
214
216
219
221
223
225
228
230
232
233
234
235
236
237
238
239
240
241
242
243
244
245
246
247
248
252
254
255
262
273
275
276
277
279
285
288
298
300
301
302
304
306
312
315
316
324
326
328
330
343
352
364
365
367
369
378
383
385
390
391
393
409
410
411
414
417
418
419
420
425
426
442
444
445
447
448
459
460
462
467
469
473
475
479
480
483
486
487
488
497
500
501
505
506
516
518
523
524
525
526
528
529
530
532
535
538
539
540
541
543
548
549
553
555
556
568
573
579
587
589
590
594
599
601
603
607
608
611
613
615
621
625
626
629
631
632
633
634
635
636
637
638
639
640
641
642
643
644
645
647
651
657
661
663
666
673
674
676
678
684
686
687
690
693
696
697
698
700
702
703
707
708
709
711
716
718
721
723
724
725
729
736
739
742
743
746
750
752
757
766
767
769
777
783
788
793
795
803
809
812
813
817
820
825
835
836
837
840
845
850
853
866
868
871
874
879
908
909
919
927
929
933
935
939
941
942
944
953
956
957
959
967
973
975
980
987
993
995
997
1003
1008
1011
1017
1018
1019
1020
1023
1034
1041
1045
1050
1052
1054
1056
1060
1071
1073
1076
1080
1089
1091
1093
1094
1101
1109
1112
1115
1118
1120
1123
1130
1132
1133
1137
1141
1144
1147
1149
1154
1164
1167
1169
1174
1175
1177
1185
1190
1194
1199
1205
1212
1231
1235
1240
1248
1249
1254
1257
1262
1270
1279
1294
1295
1297
1302
1310
1315
1328
1334
1335
1340
1341
1344
1351
1356
1360
1367
1376
1380
1387
1393
1400
1416
1420
1423
1424
1425
1431
1433
1437
1438
1442
1452
1454
1460
1464
1468
1481
1488
1489
1492
1494
1508
1512
1520
1537
1541
1543
1547
1549
1555
1567
1577
1578
1606
1613
1623
1629
1633
1634
1635
1636
1637
1638
1660
1661
1671
1674
1681
1682
1687
1693
1697
1698
1699
1701
1722
1725
1739
1742
1748
1762
1764
1767
1769
1786
1798
1810
1823
1855
1859
1860
1872
1905
1915
1938
1942
1952
1958
1959
1998
2129
\.



COPY "SubjectArea" ("ID") FROM stdin;
\.



COPY "User" ("ID") FROM stdin;
1
3
5
9
11
12
13
14
16
17
22
24
27
28
31
32
33
34
35
40
44
45
46
47
48
50
51
52
53
54
56
57
64
67
68
69
70
71
74
76
78
80
82
83
84
85
86
88
91
92
93
94
96
97
101
103
104
106
107
108
109
110
116
122
128
138
145
146
148
151
157
158
161
162
163
164
165
168
170
172
179
180
181
183
185
186
187
188
189
191
195
196
197
198
199
200
201
203
205
209
214
216
219
221
223
224
225
226
228
230
232
233
234
235
236
237
238
239
240
241
242
243
244
245
246
247
248
249
252
254
255
262
273
275
276
277
278
279
281
285
286
287
288
290
292
298
300
301
302
303
304
306
307
312
315
316
324
326
328
329
330
331
332
338
339
343
344
348
352
361
362
363
364
365
367
369
370
372
376
377
378
379
383
385
390
391
392
393
404
409
410
411
412
414
416
417
418
419
420
421
425
426
438
441
442
444
445
446
447
448
459
460
461
462
463
467
469
473
475
476
477
479
480
483
486
487
488
489
497
500
501
505
506
507
508
509
513
516
518
523
524
525
526
528
529
530
531
532
535
537
538
539
540
541
543
544
548
549
550
551
553
555
556
566
568
573
579
587
588
589
590
591
594
597
599
601
603
604
607
608
609
610
611
613
615
621
625
626
627
629
630
631
632
633
634
635
636
637
638
639
640
641
642
643
644
645
647
651
657
661
662
663
666
670
671
672
673
674
676
678
684
686
687
688
690
693
695
696
697
698
700
702
703
707
708
709
711
713
716
718
721
722
723
724
725
729
730
736
739
742
743
744
745
746
750
752
757
762
766
767
768
769
777
783
784
788
789
790
793
795
796
798
800
803
805
809
812
813
814
815
817
820
822
825
826
830
831
835
836
837
838
840
845
846
850
851
852
853
861
866
867
868
869
871
874
879
885
901
907
908
909
910
911
912
919
927
928
929
933
935
939
940
941
942
943
944
953
956
957
959
964
967
973
975
980
981
982
985
986
987
992
993
994
995
997
1000
1003
1008
1009
1011
1016
1017
1018
1019
1020
1023
1025
1034
1036
1041
1042
1045
1046
1050
1052
1054
1056
1060
1062
1070
1071
1072
1073
1074
1076
1080
1083
1089
1091
1092
1093
1094
1095
1100
1101
1102
1103
1106
1109
1112
1115
1116
1118
1120
1123
1124
1125
1130
1132
1133
1137
1141
1144
1145
1147
1149
1154
1157
1164
1167
1169
1174
1175
1177
1180
1185
1189
1190
1191
1194
1199
1205
1211
1212
1221
1222
1231
1235
1236
1238
1240
1247
1248
1249
1254
1257
1262
1270
1271
1276
1277
1279
1280
1289
1294
1295
1297
1302
1305
1308
1310
1315
1327
1328
1331
1333
1334
1335
1337
1340
1341
1342
1343
1344
1350
1351
1355
1356
1357
1360
1361
1362
1367
1376
1378
1380
1387
1392
1393
1395
1399
1400
1401
1405
1406
1407
1416
1418
1419
1420
1422
1423
1424
1425
1431
1433
1436
1437
1438
1439
1442
1446
1449
1452
1454
1460
1464
1467
1468
1469
1473
1477
1478
1479
1481
1488
1489
1492
1494
1503
1506
1507
1508
1512
1514
1515
1517
1520
1522
1537
1541
1542
1543
1544
1547
1549
1553
1555
1561
1562
1564
1567
1568
1571
1577
1578
1582
1591
1605
1606
1607
1613
1617
1618
1623
1624
1629
1633
1634
1635
1636
1637
1638
1642
1651
1654
1660
1661
1671
1674
1680
1681
1682
1687
1693
1696
1697
1698
1699
1701
1705
1706
1709
1715
1722
1725
1733
1738
1739
1742
1745
1748
1755
1761
1762
1763
1764
1767
1769
1773
1775
1786
1792
1798
1800
1801
1810
1823
1830
1839
1840
1844
1845
1846
1847
1851
1852
1855
1859
1860
1861
1872
1878
1882
1890
1895
1901
1904
1905
1910
1915
1919
1938
1942
1944
1952
1955
1958
1959
1979
1992
1994
1998
2001
2006
2016
2017
2022
2025
2026
2029
2038
2050
2053
2056
2061
2066
2069
2071
2073
2090
2097
2115
2118
2124
2129
2151
2158
2170
2173
2182
2210
2212
2217
\.



COPY "assignReviewer" ("Administrator", "Reviewer") FROM stdin;
\.



COPY "assignedTo" ("Paper", "Reviewer") FROM stdin;
10	12
10	13
19	17
19	1302
29	1018
29	1769
30	33
30	34
37	106
37	109
38	35
38	1958
42	40
42	693
43	45
43	46
49	51
49	53
59	57
59	641
60	57
60	641
62	589
62	590
66	67
66	71
81	83
81	84
90	91
90	92
98	483
98	553
105	587
105	1376
111	3
111	1481
112	199
112	201
113	199
113	201
114	3
114	1481
115	24
115	200
117	316
117	729
118	645
118	987
119	645
119	987
120	116
120	603
123	835
123	836
124	47
124	1045
126	736
126	1093
127	45
127	46
129	151
129	462
132	128
132	1454
133	128
133	1454
134	151
134	462
140	1
140	1310
141	70
141	138
143	70
143	138
147	1141
147	1356
149	635
149	663
150	635
150	663
153	459
153	813
154	459
154	813
159	529
159	532
160	161
160	163
174	172
174	1177
178	101
178	180
184	187
184	189
192	1089
192	1508
194	196
194	197
207	205
207	1512
211	56
211	209
212	769
212	1547
218	162
218	219
222	223
222	225
250	300
250	690
251	219
251	252
256	255
256	312
257	528
257	633
259	393
259	644
260	393
260	644
263	639
263	687
264	626
264	1050
265	236
265	262
267	236
267	262
269	639
269	687
270	78
270	262
274	277
274	279
282	723
282	724
283	523
283	525
284	285
284	288
291	556
291	997
294	276
294	634
295	276
295	634
299	302
299	304
308	288
308	1054
309	288
309	1054
310	275
310	1633
319	316
319	673
320	316
320	673
327	328
327	330
333	632
333	1577
334	246
334	417
335	246
335	417
340	225
340	1433
341	168
341	1164
345	500
345	953
346	500
346	845
347	500
347	953
349	767
349	1112
350	64
350	956
351	64
351	956
353	352
353	1091
354	1094
354	1959
355	352
355	383
356	76
356	1671
359	76
359	1671
360	364
360	365
371	757
371	1492
373	1638
373	1855
374	1248
374	1249
375	214
375	378
380	445
380	1661
381	306
381	1297
387	385
387	803
394	145
394	927
395	16
395	1351
396	33
396	34
397	16
397	1351
398	16
398	390
400	315
400	640
401	232
401	425
402	232
402	425
405	516
405	1762
408	409
408	411
415	418
415	419
422	959
422	1823
429	240
429	941
432	426
432	942
433	240
433	941
435	426
435	783
436	426
436	783
440	22
440	442
443	444
443	445
449	448
449	611
450	214
450	1118
453	448
453	1423
454	247
454	1674
456	247
456	1674
457	448
457	611
458	460
458	462
464	233
464	647
465	233
465	647
474	275
474	475
482	480
482	853
485	486
485	487
492	414
492	1660
493	444
493	445
503	276
503	501
504	505
504	506
510	599
510	1606
511	235
511	1942
512	993
512	1212
514	548
514	549
521	636
521	1120
522	636
522	1120
536	539
536	540
545	793
545	1334
558	324
558	1270
559	487
559	539
562	488
562	1231
563	487
563	539
567	939
567	1425
569	188
569	568
571	188
571	568
575	573
575	1344
581	170
581	777
582	579
582	1120
583	661
583	1132
592	637
592	840
596	594
596	1060
606	1
606	607
617	615
617	666
618	479
618	850
623	157
623	621
624	625
624	626
628	326
628	631
653	651
653	1041
659	657
659	1872
669	673
669	674
685	686
685	687
689	1076
689	1701
694	696
694	697
705	703
705	1549
706	707
706	709
714	96
714	1623
727	50
727	725
732	230
732	1073
733	230
733	1073
734	162
734	164
741	742
741	746
754	752
754	980
755	44
755	1315
756	45
756	1613
763	1130
763	1167
765	766
765	767
771	769
771	1488
773	1489
773	1786
775	769
775	1488
776	56
776	795
778	777
778	1008
781	641
781	711
785	543
785	1340
787	200
787	788
791	820
791	1488
801	88
801	739
802	88
802	739
806	1023
806	1056
811	22
811	812
824	31
824	237
827	929
827	1147
832	239
832	1212
833	239
833	1212
847	1328
847	1722
856	200
856	788
857	367
857	1279
864	107
864	1725
865	866
865	868
876	874
876	1341
881	879
881	1400
883	221
883	739
888	240
888	608
891	1
891	607
893	71
893	367
894	71
894	367
895	24
895	69
896	24
896	69
903	103
903	201
906	908
906	909
913	935
913	1567
914	643
914	1199
915	643
915	1199
918	85
918	919
921	244
921	1380
922	678
922	766
923	766
923	767
924	244
924	1380
926	238
926	390
946	944
946	1682
947	9
947	608
949	9
949	608
965	721
965	1442
969	967
969	1637
976	228
976	1137
978	157
978	254
979	157
979	254
991	993
991	995
1010	1154
1010	1698
1013	1011
1013	1636
1015	1017
1015	1019
1021	326
1021	1295
1029	85
1029	1635
1030	708
1030	1952
1032	85
1032	919
1038	684
1038	1416
1049	78
1049	273
1051	245
1051	1052
1057	676
1057	795
1058	391
1058	393
1063	216
1063	420
1065	418
1065	1034
1066	216
1066	420
1069	447
1069	1071
1082	473
1082	927
1087	418
1087	419
1090	1093
1090	1094
1098	203
1098	417
1099	243
1099	1101
1111	1109
1111	1520
1117	80
1117	1767
1122	82
1122	181
1127	12
1127	13
1135	1133
1135	2129
1139	637
1139	1185
1140	637
1140	1185
1143	837
1143	1144
1151	524
1151	1149
1156	74
1156	750
1159	183
1159	1169
1160	473
1160	1555
1161	183
1161	1169
1173	84
1173	497
1181	80
1181	1115
1182	326
1182	631
1184	629
1184	1294
1188	541
1188	1190
1192	76
1192	410
1196	684
1196	1393
1197	933
1197	1194
1198	684
1198	1393
1200	223
1200	225
1210	640
1210	1080
1218	1091
1218	1235
1219	1093
1219	1094
1225	185
1225	188
1226	187
1226	189
1229	223
1229	1748
1239	1335
1239	1699
1242	469
1242	1240
1243	601
1243	709
1244	601
1244	709
1250	82
1250	1175
1252	82
1252	1175
1256	555
1256	1254
1259	530
1259	1262
1260	1257
1260	1452
1268	635
1268	1634
1272	275
1272	475
1281	973
1281	1420
1298	22
1298	442
1307	638
1307	1205
1317	245
1317	813
1324	163
1324	750
1325	74
1325	750
1326	640
1326	1080
1352	33
1352	825
1359	953
1359	1360
1363	625
1363	626
1364	613
1364	1387
1366	613
1366	1387
1369	1020
1369	1367
1372	698
1372	743
1377	817
1377	1008
1384	234
1384	535
1385	234
1385	535
1390	1464
1390	1468
1398	642
1398	1167
1404	645
1404	716
1435	1437
1435	1438
1440	1810
1440	1905
1443	221
1443	241
1445	221
1445	241
1450	301
1450	871
1476	632
1476	957
1483	993
1483	995
1484	975
1484	1199
1518	1118
1518	1424
1531	638
1531	1205
1532	1123
1532	1764
1533	414
1533	1297
1534	1205
1534	1687
1539	467
1539	1537
1540	1541
1540	1543
1554	27
1554	248
1557	633
1557	1003
1558	633
1558	1003
1559	809
1559	1494
1576	203
1576	1578
1585	5
1585	526
1587	245
1587	1052
1588	442
1588	1742
1596	78
1596	273
1609	518
1609	538
1616	538
1616	1231
1619	1859
1619	1860
1620	599
1620	1915
1621	700
1621	1798
1631	1629
1631	1697
1646	242
1646	369
1647	242
1647	369
1662	161
1662	163
1667	1393
1667	1998
1669	702
1669	1115
1695	298
1695	1693
1704	718
1704	1431
1710	475
1710	1938
1714	83
1714	1681
1729	203
1729	1578
1730	632
1730	957
1737	1460
1737	1739
1751	162
1751	219
1757	1174
1757	1175
1771	642
1771	1167
1883	238
1883	390
1889	343
1889	611
1899	27
1899	248
2009	473
2009	927
2102	31
2102	237
\.



COPY "co-writePaper" ("Co-author", "Paper") FROM stdin;
1	947
1	949
3	112
3	113
3	1484
11	10
11	1127
12	251
12	263
12	269
13	685
16	30
16	394
16	396
22	1051
22	1587
24	893
24	894
27	394
31	30
31	396
31	458
33	398
33	824
33	2102
35	37
44	43
44	127
44	284
46	755
47	43
47	126
47	127
50	49
51	727
54	49
57	62
64	333
67	1384
67	1385
69	66
70	66
70	606
70	891
71	115
71	1384
71	1385
76	353
76	408
78	624
78	1363
80	628
80	1182
80	1667
82	81
82	178
83	918
83	1032
83	1757
84	1250
84	1252
85	81
85	1030
88	1225
91	98
92	504
93	90
96	422
97	98
103	111
103	114
107	37
116	117
116	118
116	119
128	129
128	134
138	140
145	926
145	1883
146	147
148	149
148	150
151	153
151	154
158	159
161	251
162	160
162	1662
163	218
163	1751
164	160
164	1662
181	178
183	435
183	436
183	1160
185	184
185	1226
188	184
188	883
188	1226
189	400
189	569
189	571
191	192
196	582
197	521
197	522
198	194
199	787
199	856
200	112
200	113
200	857
201	111
201	114
201	115
203	1065
203	1476
203	1730
209	212
214	1243
214	1244
216	1065
219	732
219	733
219	1324
225	881
226	222
226	1200
228	592
228	978
228	979
230	734
232	1210
232	1225
232	1326
233	811
235	510
238	1160
239	1620
240	140
240	435
240	436
244	765
244	923
245	440
245	1298
247	449
247	457
248	1082
248	2009
252	732
252	733
254	976
255	257
255	259
255	260
262	263
262	264
262	269
275	274
276	274
279	294
279	295
287	284
290	291
292	174
298	174
300	299
301	299
302	754
307	308
307	309
307	310
315	401
315	402
326	1181
326	1184
329	327
338	291
339	340
343	1518
352	354
352	356
352	359
361	360
363	360
367	895
367	896
369	903
369	914
369	915
370	371
376	375
378	706
391	256
393	256
393	776
404	405
409	356
409	359
409	1021
410	408
414	443
414	493
414	1534
417	333
417	415
417	1087
418	334
418	335
419	1098
420	415
420	1087
426	429
426	433
442	464
442	465
442	1317
444	1069
445	492
447	212
447	443
447	493
448	450
448	454
448	456
459	458
460	132
460	133
462	132
462	133
462	1317
467	806
473	398
473	1554
473	1899
476	474
476	1272
477	105
486	1188
487	1188
487	1616
488	485
497	1122
500	583
508	504
513	514
518	257
524	283
525	282
526	283
528	1559
530	159
538	536
538	1557
538	1558
539	562
540	562
541	536
544	545
550	514
556	558
566	567
573	341
579	581
589	781
590	59
590	60
590	1038
597	387
599	511
599	512
603	319
603	320
604	250
607	888
608	432
608	606
608	891
611	1518
613	453
613	624
613	1363
625	1364
625	1366
626	1049
626	1596
629	628
629	1182
629	1192
632	1576
632	1729
633	1559
633	1616
635	583
636	778
638	1533
639	270
640	401
640	402
640	1324
641	1377
645	120
661	149
661	150
661	1359
670	669
671	669
671	978
671	979
673	120
676	1058
678	1443
678	1445
684	1181
684	1197
686	265
686	267
687	265
687	267
688	689
693	727
693	1620
698	694
703	374
707	1242
708	706
708	918
708	1032
709	450
709	1242
711	1377
722	282
725	42
736	1737
739	569
739	571
739	922
743	741
745	741
750	734
750	1210
750	1326
762	763
766	1443
766	1445
767	883
767	921
767	924
769	773
769	776
777	521
777	522
777	781
783	1159
783	1161
788	903
793	1435
795	259
795	260
795	791
798	785
800	801
800	802
803	374
812	1051
812	1587
813	129
813	134
813	811
820	211
822	617
825	824
825	2102
826	827
837	123
838	123
846	847
850	482
851	482
861	617
867	865
885	596
901	207
910	906
911	906
912	913
919	1714
927	926
927	1883
933	1667
939	1135
942	429
942	433
956	349
956	1476
956	1730
957	350
957	351
975	1646
975	1647
981	754
985	118
985	119
985	503
993	914
993	915
994	991
994	1483
995	1484
1003	1609
1008	59
1008	60
1008	582
1018	1015
1020	1015
1025	141
1025	143
1034	334
1034	335
1036	864
1042	207
1045	756
1050	1049
1050	1596
1052	1588
1062	503
1074	659
1080	1156
1080	1325
1091	1090
1091	1219
1093	353
1093	1737
1094	1218
1095	623
1102	1099
1103	1099
1112	921
1112	924
1116	1117
1120	581
1123	1122
1123	1307
1123	1531
1132	345
1132	347
1137	623
1137	1139
1137	1140
1145	1143
1147	1135
1167	881
1174	1173
1175	1173
1194	1196
1194	1198
1199	512
1199	1646
1199	1647
1205	381
1222	340
1222	1139
1222	1140
1231	559
1231	563
1235	126
1240	1029
1240	1243
1240	1244
1247	1013
1249	373
1257	1259
1270	1619
1271	596
1276	1256
1279	895
1279	896
1280	1281
1289	1057
1295	1184
1297	380
1297	1307
1297	1531
1310	947
1310	949
1315	308
1315	309
1327	38
1331	346
1334	1390
1343	876
1344	876
1350	946
1351	1352
1355	147
1357	19
1362	1359
1376	618
1380	922
1387	264
1393	1669
1400	1229
1400	1398
1400	1771
1401	1268
1401	1398
1401	1771
1405	1404
1407	294
1407	295
1407	1404
1416	1197
1419	1281
1420	1539
1423	454
1423	456
1424	453
1439	1440
1467	1390
1469	705
1478	422
1479	105
1488	211
1489	771
1489	775
1494	1557
1494	1558
1503	395
1503	397
1506	192
1508	1695
1515	567
1520	618
1542	1540
1547	773
1555	1554
1555	1899
1561	965
1564	1010
1571	387
1577	1576
1577	1729
1582	1369
1591	969
1605	832
1605	833
1606	832
1606	833
1624	714
1635	1030
1636	558
1642	1063
1642	1066
1661	492
1671	355
1674	1889
1681	1029
1687	381
1697	1695
1706	1704
1722	1539
1733	1239
1739	354
1742	464
1742	465
1755	1372
1763	405
1764	1534
1767	1021
1773	785
1786	791
1792	1268
1801	1621
1830	1532
1839	847
1845	1013
1847	310
1851	29
1852	29
1861	1710
1882	1450
1890	1631
1904	1440
1905	1259
1910	913
1915	511
1944	1151
1959	1218
1979	341
1992	19
1998	1669
2001	689
2016	1229
2022	1111
2026	1585
2029	1010
2053	1450
2066	124
2069	1239
2071	653
2090	1369
2097	864
2115	1260
2129	969
2151	1585
2170	38
2173	575
2210	373
\.



COPY "detailsEnteredBy" ("Conference", "Administrator") FROM stdin;
\.



COPY "enableVirtualMeeting" ("Administrator", "Conference") FROM stdin;
\.



COPY "endReview" ("ProgramCommitteeChair", "Review") FROM stdin;
\.



COPY "enterReviewCriteria" ("Administrator", "Conference") FROM stdin;
\.



COPY "finalizePaperAssignment" ("Administrator", "Conference") FROM stdin;
\.



COPY "hardcopyMailingManifestsP" ("Conference", "Administrator") FROM stdin;
\.



COPY "hasConferenceMember" ("Conference", "ConferenceMember") FROM stdin;
20	17
20	56
20	83
20	106
20	116
20	200
20	221
20	277
20	298
20	302
20	352
20	364
20	390
20	467
20	516
20	523
20	538
20	548
20	594
20	661
20	666
20	750
20	813
20	850
20	879
20	909
20	1019
20	1076
20	1115
20	1118
20	1133
20	1141
20	1212
20	1248
20	1262
20	1270
20	1340
20	1512
20	1567
20	1698
20	1699
130	3
130	9
130	24
130	78
130	84
130	101
130	128
130	163
130	170
130	187
130	203
130	209
130	230
130	232
130	233
130	234
130	236
130	237
130	238
130	239
130	240
130	241
130	242
130	244
130	245
130	246
130	247
130	248
130	306
130	312
130	409
130	414
130	469
130	541
130	702
130	708
130	1008
130	1091
130	1194
130	1231
130	1294
130	1424
130	1489
130	1494
130	1739
136	3
136	9
136	24
136	51
136	78
136	128
136	163
136	187
136	203
136	230
136	232
136	233
136	234
136	235
136	236
136	237
136	238
136	239
136	240
136	241
136	242
136	243
136	244
136	245
136	246
136	247
136	248
175	1
175	5
175	107
175	157
175	164
175	172
175	188
175	276
175	326
175	385
175	418
175	483
175	540
175	709
175	742
175	752
175	812
175	853
175	939
175	1018
175	1120
175	1164
175	1199
175	1205
175	1360
175	1367
175	1423
175	1433
175	1468
175	1547
175	1555
175	1633
175	1905
175	1915
175	1958
175	1959
427	34
427	46
427	76
427	85
427	201
427	254
427	288
427	419
427	426
427	445
427	462
427	568
427	601
427	607
427	626
427	631
427	632
427	633
427	634
427	635
427	636
427	637
427	638
427	639
427	640
427	641
427	642
427	643
427	644
427	645
430	34
430	46
430	50
430	76
430	85
430	201
430	254
430	288
430	419
430	426
430	445
430	462
430	568
430	601
430	607
430	626
430	631
430	632
430	633
430	634
430	635
430	636
430	637
430	638
430	639
430	640
430	641
430	642
430	643
430	644
430	645
665	71
665	82
665	103
665	219
665	223
665	285
665	460
665	505
665	528
665	532
665	673
665	711
665	716
665	724
665	766
665	836
665	871
665	927
665	959
665	997
665	1050
665	1093
665	1328
665	1341
665	1376
665	1393
665	1420
665	1488
665	1508
665	1577
665	1634
665	1635
665	1636
665	1637
665	1638
\.



COPY "hasConflictOfInterest" ("Person", "Document") FROM stdin;
\.



COPY "hasProgramCommitteeMember" ("ProgramCommittee", "ProgramCommitteeMember") FROM stdin;
318	316
318	393
318	1034
318	1367
318	1810
565	24
565	31
565	241
565	273
668	84
668	200
668	228
668	523
668	666
668	743
668	1701
1005	88
1005	247
1005	411
1005	459
1005	535
1005	541
1005	1297
1266	45
1266	219
1266	473
1266	635
1266	644
1267	45
1267	219
1267	473
1267	635
1267	644
1330	487
1330	637
1330	678
1330	1089
1330	1328
\.



COPY "hasSubjectArea" ("Paper", "SubjectArea") FROM stdin;
\.



COPY "markConflictOfInterest" ("ConferenceMember", "Paper") FROM stdin;
\.



COPY "paperAssignmentToolsRunBy" ("Conference", "Administrator") FROM stdin;
\.



COPY "readByReviewer" ("Paper", "Reviewer") FROM stdin;
\.



COPY "reviewerBiddingStartedBy" ("Conference", "Administrator") FROM stdin;
\.



COPY "setMaxPapers" ("Administrator", "ProgramCommitteeMember") FROM stdin;
\.



COPY "submitPaper" ("Author", "Paper") FROM stdin;
\.



ALTER TABLE ONLY "Acceptance"
    ADD CONSTRAINT "AcceptancePK" PRIMARY KEY ("ID");



ALTER TABLE ONLY "Administrator"
    ADD CONSTRAINT "AdministratorPK" PRIMARY KEY ("ID");



ALTER TABLE ONLY "AssociatedChair"
    ADD CONSTRAINT "AssociatedChairPK" PRIMARY KEY ("ID");



ALTER TABLE ONLY "AuthorNotReviewer"
    ADD CONSTRAINT "AuthorNotReviewerPK" PRIMARY KEY ("ID");



ALTER TABLE ONLY "Author"
    ADD CONSTRAINT "AuthorPK" PRIMARY KEY ("ID");



ALTER TABLE ONLY "Bid"
    ADD CONSTRAINT "BidPK" PRIMARY KEY ("ID");



ALTER TABLE ONLY "Chairman"
    ADD CONSTRAINT "ChairmanPK" PRIMARY KEY ("ID");



ALTER TABLE ONLY "Co-author"
    ADD CONSTRAINT "Co-authorPK" PRIMARY KEY ("ID");



ALTER TABLE ONLY "ConferenceChair"
    ADD CONSTRAINT "ConferenceChairPK" PRIMARY KEY ("ID");



ALTER TABLE ONLY "ConferenceMember"
    ADD CONSTRAINT "ConferenceMemberPK" PRIMARY KEY ("ID");



ALTER TABLE ONLY "Conference"
    ADD CONSTRAINT "ConferencePK" PRIMARY KEY ("ID");



ALTER TABLE ONLY "Decision"
    ADD CONSTRAINT "DecisionPK" PRIMARY KEY ("ID");



ALTER TABLE ONLY "Document"
    ADD CONSTRAINT "DocumentPK" PRIMARY KEY ("ID");



ALTER TABLE ONLY "ExternalReviewer"
    ADD CONSTRAINT "ExternalReviewerPK" PRIMARY KEY ("ID");



ALTER TABLE ONLY "Meta-Review"
    ADD CONSTRAINT "Meta-ReviewPK" PRIMARY KEY ("ID");



ALTER TABLE ONLY "Meta-Reviewer"
    ADD CONSTRAINT "Meta-ReviewerPK" PRIMARY KEY ("ID");



ALTER TABLE ONLY "PaperAbstract"
    ADD CONSTRAINT "PaperAbstractPK" PRIMARY KEY ("ID");



ALTER TABLE ONLY "PaperFullVersion"
    ADD CONSTRAINT "PaperFullVersionPK" PRIMARY KEY ("ID");



ALTER TABLE ONLY "Paper"
    ADD CONSTRAINT "PaperPK" PRIMARY KEY ("ID");



ALTER TABLE ONLY "Person"
    ADD CONSTRAINT "PersonPK" PRIMARY KEY ("ID");



ALTER TABLE ONLY "Preference"
    ADD CONSTRAINT "PreferencePK" PRIMARY KEY ("ID");



ALTER TABLE ONLY "ProgramCommitteeChair"
    ADD CONSTRAINT "ProgramCommitteeChairPK" PRIMARY KEY ("ID");



ALTER TABLE ONLY "ProgramCommitteeMember"
    ADD CONSTRAINT "ProgramCommitteeMemberPK" PRIMARY KEY ("ID");



ALTER TABLE ONLY "ProgramCommittee"
    ADD CONSTRAINT "ProgramCommitteePK" PRIMARY KEY ("ID");



ALTER TABLE ONLY "Rejection"
    ADD CONSTRAINT "RejectionPK" PRIMARY KEY ("ID");



ALTER TABLE ONLY "Review"
    ADD CONSTRAINT "ReviewPK" PRIMARY KEY ("ID");



ALTER TABLE ONLY "Reviewer"
    ADD CONSTRAINT "ReviewerPK" PRIMARY KEY ("ID");



ALTER TABLE ONLY "SubjectArea"
    ADD CONSTRAINT "SubjectAreaPK" PRIMARY KEY ("ID");



ALTER TABLE ONLY "User"
    ADD CONSTRAINT "UserPK" PRIMARY KEY ("ID");



ALTER TABLE ONLY "assignReviewer"
    ADD CONSTRAINT "assignReviewerPK" PRIMARY KEY ("Administrator", "Reviewer");



ALTER TABLE ONLY "assignedTo"
    ADD CONSTRAINT "assignedToPK" PRIMARY KEY ("Paper", "Reviewer");



ALTER TABLE ONLY "co-writePaper"
    ADD CONSTRAINT "co-writePaperPK" PRIMARY KEY ("Co-author", "Paper");



ALTER TABLE ONLY "detailsEnteredBy"
    ADD CONSTRAINT "detailsEnteredByPK" PRIMARY KEY ("Conference", "Administrator");



ALTER TABLE ONLY "enableVirtualMeeting"
    ADD CONSTRAINT "enableVirtualMeetingPK" PRIMARY KEY ("Administrator", "Conference");



ALTER TABLE ONLY "endReview"
    ADD CONSTRAINT "endReviewPK" PRIMARY KEY ("ProgramCommitteeChair", "Review");



ALTER TABLE ONLY "enterReviewCriteria"
    ADD CONSTRAINT "enterReviewCriteriaPK" PRIMARY KEY ("Administrator", "Conference");



ALTER TABLE ONLY "finalizePaperAssignment"
    ADD CONSTRAINT "finalizePaperAssignmentPK" PRIMARY KEY ("Administrator", "Conference");



ALTER TABLE ONLY "hardcopyMailingManifestsP"
    ADD CONSTRAINT "hardcopyMailingManifestsPPK" PRIMARY KEY ("Conference", "Administrator");



ALTER TABLE ONLY "hasConferenceMember"
    ADD CONSTRAINT "hasConferenceMemberPK" PRIMARY KEY ("Conference", "ConferenceMember");



ALTER TABLE ONLY "hasConflictOfInterest"
    ADD CONSTRAINT "hasConflictOfInterestPK" PRIMARY KEY ("Person", "Document");



ALTER TABLE ONLY "hasProgramCommitteeMember"
    ADD CONSTRAINT "hasProgramCommitteeMemberPK" PRIMARY KEY ("ProgramCommittee", "ProgramCommitteeMember");



ALTER TABLE ONLY "hasSubjectArea"
    ADD CONSTRAINT "hasSubjectAreaPK" PRIMARY KEY ("Paper", "SubjectArea");



ALTER TABLE ONLY "markConflictOfInterest"
    ADD CONSTRAINT "markConflictOfInterestPK" PRIMARY KEY ("ConferenceMember", "Paper");



ALTER TABLE ONLY "paperAssignmentToolsRunBy"
    ADD CONSTRAINT "paperAssignmentToolsRunByPK" PRIMARY KEY ("Conference", "Administrator");



ALTER TABLE ONLY "readByReviewer"
    ADD CONSTRAINT "readByReviewerPK" PRIMARY KEY ("Paper", "Reviewer");



ALTER TABLE ONLY "reviewerBiddingStartedBy"
    ADD CONSTRAINT "reviewerBiddingStartedByPK" PRIMARY KEY ("Conference", "Administrator");



ALTER TABLE ONLY "setMaxPapers"
    ADD CONSTRAINT "setMaxPapersPK" PRIMARY KEY ("Administrator", "ProgramCommitteeMember");



ALTER TABLE ONLY "submitPaper"
    ADD CONSTRAINT "submitPaperPK" PRIMARY KEY ("Author", "Paper");



ALTER TABLE ONLY "Acceptance"
    ADD CONSTRAINT "FKAcceptanceURI" FOREIGN KEY ("ID") REFERENCES "Decision"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "Administrator"
    ADD CONSTRAINT "FKAdministratorURI" FOREIGN KEY ("ID") REFERENCES "User"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "Administrator"
    ADD CONSTRAINT "FKAdministratoracceptPaper" FOREIGN KEY ("acceptPaper") REFERENCES "Paper"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "AssociatedChair"
    ADD CONSTRAINT "FKAssociatedChairURI" FOREIGN KEY ("ID") REFERENCES "Chairman"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "AuthorNotReviewer"
    ADD CONSTRAINT "FKAuthorNotReviewerURI" FOREIGN KEY ("ID") REFERENCES "Author"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "Author"
    ADD CONSTRAINT "FKAuthorURI" FOREIGN KEY ("ID") REFERENCES "ConferenceMember"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "Author"
    ADD CONSTRAINT "FKAuthorURI2" FOREIGN KEY ("ID") REFERENCES "User"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "Bid"
    ADD CONSTRAINT "FKBidadjustedBy" FOREIGN KEY ("adjustedBy") REFERENCES "Reviewer"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "Bid"
    ADD CONSTRAINT "FKBidhasBid_Inv" FOREIGN KEY ("hasBid_Inv") REFERENCES "Paper"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "Chairman"
    ADD CONSTRAINT "FKChairmanURI" FOREIGN KEY ("ID") REFERENCES "ConferenceMember"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "Co-author"
    ADD CONSTRAINT "FKCo-authorURI" FOREIGN KEY ("ID") REFERENCES "Author"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "ConferenceChair"
    ADD CONSTRAINT "FKConferenceChairURI" FOREIGN KEY ("ID") REFERENCES "Chairman"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "ConferenceMember"
    ADD CONSTRAINT "FKConferenceMemberURI" FOREIGN KEY ("ID") REFERENCES "Person"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "ExternalReviewer"
    ADD CONSTRAINT "FKExternalReviewerURI" FOREIGN KEY ("ID") REFERENCES "Person"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "ExternalReviewer"
    ADD CONSTRAINT "FKExternalReviewerassigne" FOREIGN KEY ("assignedByReviewer") REFERENCES "Reviewer"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "Meta-Review"
    ADD CONSTRAINT "FKMeta-ReviewURI" FOREIGN KEY ("ID") REFERENCES "Review"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "Meta-Reviewer"
    ADD CONSTRAINT "FKMeta-ReviewerURI" FOREIGN KEY ("ID") REFERENCES "Reviewer"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "PaperAbstract"
    ADD CONSTRAINT "FKPaperAbstractURI" FOREIGN KEY ("ID") REFERENCES "Paper"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "PaperFullVersion"
    ADD CONSTRAINT "FKPaperFullVersionURI" FOREIGN KEY ("ID") REFERENCES "Paper"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "Paper"
    ADD CONSTRAINT "FKPaperURI" FOREIGN KEY ("ID") REFERENCES "Document"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "Paper"
    ADD CONSTRAINT "FKPaperacceptedBy" FOREIGN KEY ("acceptedBy") REFERENCES "Administrator"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "Paper"
    ADD CONSTRAINT "FKPaperhasAuthor" FOREIGN KEY ("hasAuthor") REFERENCES "Author"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "Paper"
    ADD CONSTRAINT "FKPaperhasDecision" FOREIGN KEY ("hasDecision") REFERENCES "Decision"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "Paper"
    ADD CONSTRAINT "FKPaperreadByMeta-Reviewer" FOREIGN KEY ("readByMeta-Reviewer") REFERENCES "Meta-Reviewer"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "Paper"
    ADD CONSTRAINT "FKPaperrejectedBy" FOREIGN KEY ("rejectedBy") REFERENCES "Administrator"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "ProgramCommitteeChair"
    ADD CONSTRAINT "FKProgramCommitteeChairURI" FOREIGN KEY ("ID") REFERENCES "Chairman"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "ProgramCommitteeChair"
    ADD CONSTRAINT "FKProgramCommitteeChairURI2" FOREIGN KEY ("ID") REFERENCES "ProgramCommitteeMember"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "ProgramCommitteeMember"
    ADD CONSTRAINT "FKProgramCommitteeMemberURI" FOREIGN KEY ("ID") REFERENCES "ConferenceMember"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "ProgramCommitteeMember"
    ADD CONSTRAINT "FKProgramCommitteeMembera" FOREIGN KEY ("addedBy") REFERENCES "Administrator"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "Rejection"
    ADD CONSTRAINT "FKRejectionURI" FOREIGN KEY ("ID") REFERENCES "Decision"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "Review"
    ADD CONSTRAINT "FKReviewURI" FOREIGN KEY ("ID") REFERENCES "Document"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "Reviewer"
    ADD CONSTRAINT "FKReviewerURI" FOREIGN KEY ("ID") REFERENCES "ConferenceMember"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "Reviewer"
    ADD CONSTRAINT "FKReviewerURI2" FOREIGN KEY ("ID") REFERENCES "User"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "Review"
    ADD CONSTRAINT "FKReviewwrittenBy" FOREIGN KEY ("writtenBy") REFERENCES "Reviewer"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "User"
    ADD CONSTRAINT "FKUserURI" FOREIGN KEY ("ID") REFERENCES "Person"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "assignReviewer"
    ADD CONSTRAINT "FKassignReviewerAdministrator" FOREIGN KEY ("Administrator") REFERENCES "Administrator"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "assignReviewer"
    ADD CONSTRAINT "FKassignReviewerReviewer" FOREIGN KEY ("Reviewer") REFERENCES "Reviewer"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "assignedTo"
    ADD CONSTRAINT "FKassignedToPaper" FOREIGN KEY ("Paper") REFERENCES "Paper"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "assignedTo"
    ADD CONSTRAINT "FKassignedToReviewer" FOREIGN KEY ("Reviewer") REFERENCES "Reviewer"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "co-writePaper"
    ADD CONSTRAINT "FKco-writePaperCo-author" FOREIGN KEY ("Co-author") REFERENCES "Co-author"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "co-writePaper"
    ADD CONSTRAINT "FKco-writePaperPaper" FOREIGN KEY ("Paper") REFERENCES "Paper"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "detailsEnteredBy"
    ADD CONSTRAINT "FKdetailsEnteredByAdminis" FOREIGN KEY ("Administrator") REFERENCES "Administrator"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "detailsEnteredBy"
    ADD CONSTRAINT "FKdetailsEnteredByConference" FOREIGN KEY ("Conference") REFERENCES "Conference"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "enableVirtualMeeting"
    ADD CONSTRAINT "FKenableVirtualMeetingAdm" FOREIGN KEY ("Administrator") REFERENCES "Administrator"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "enableVirtualMeeting"
    ADD CONSTRAINT "FKenableVirtualMeetingCon" FOREIGN KEY ("Conference") REFERENCES "Conference"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "endReview"
    ADD CONSTRAINT "FKendReviewProgramCommitt" FOREIGN KEY ("ProgramCommitteeChair") REFERENCES "ProgramCommitteeChair"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "endReview"
    ADD CONSTRAINT "FKendReviewReview" FOREIGN KEY ("Review") REFERENCES "Review"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "enterReviewCriteria"
    ADD CONSTRAINT "FKenterReviewCriteriaAdmi" FOREIGN KEY ("Administrator") REFERENCES "Administrator"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "enterReviewCriteria"
    ADD CONSTRAINT "FKenterReviewCriteriaConf" FOREIGN KEY ("Conference") REFERENCES "Conference"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "finalizePaperAssignment"
    ADD CONSTRAINT "FKfinalizePaperAssignment" FOREIGN KEY ("Administrator") REFERENCES "Administrator"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "hardcopyMailingManifestsP"
    ADD CONSTRAINT "FKhardcopyMailingManifest" FOREIGN KEY ("Conference") REFERENCES "Conference"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "hasConferenceMember"
    ADD CONSTRAINT "FKhasConferenceMemberConf" FOREIGN KEY ("Conference") REFERENCES "Conference"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "hasConflictOfInterest"
    ADD CONSTRAINT "FKhasConflictOfInterestDo" FOREIGN KEY ("Document") REFERENCES "Document"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "hasConflictOfInterest"
    ADD CONSTRAINT "FKhasConflictOfInterestPerson" FOREIGN KEY ("Person") REFERENCES "Person"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "hasProgramCommitteeMember"
    ADD CONSTRAINT "FKhasProgramCommitteeMemb" FOREIGN KEY ("ProgramCommittee") REFERENCES "ProgramCommittee"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "hasSubjectArea"
    ADD CONSTRAINT "FKhasSubjectAreaPaper" FOREIGN KEY ("Paper") REFERENCES "Paper"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "hasSubjectArea"
    ADD CONSTRAINT "FKhasSubjectAreaSubjectArea" FOREIGN KEY ("SubjectArea") REFERENCES "SubjectArea"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "markConflictOfInterest"
    ADD CONSTRAINT "FKmarkConflictOfInterestPaper" FOREIGN KEY ("Paper") REFERENCES "Paper"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "paperAssignmentToolsRunBy"
    ADD CONSTRAINT "FKpaperAssignmentToolsRun" FOREIGN KEY ("Conference") REFERENCES "Conference"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "readByReviewer"
    ADD CONSTRAINT "FKreadByReviewerPaper" FOREIGN KEY ("Paper") REFERENCES "Paper"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "readByReviewer"
    ADD CONSTRAINT "FKreadByReviewerReviewer" FOREIGN KEY ("Reviewer") REFERENCES "Reviewer"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "reviewerBiddingStartedBy"
    ADD CONSTRAINT "FKreviewerBiddingStartedB" FOREIGN KEY ("Conference") REFERENCES "Conference"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "setMaxPapers"
    ADD CONSTRAINT "FKsetMaxPapersAdministrator" FOREIGN KEY ("Administrator") REFERENCES "Administrator"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "submitPaper"
    ADD CONSTRAINT "FKsubmitPaperAuthor" FOREIGN KEY ("Author") REFERENCES "Author"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "submitPaper"
    ADD CONSTRAINT "FKsubmitPaperPaper" FOREIGN KEY ("Paper") REFERENCES "Paper"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "finalizePaperAssignment"
    ADD CONSTRAINT "finalizePaperAssignmen2FK" FOREIGN KEY ("Conference") REFERENCES "Conference"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "hardcopyMailingManifestsP"
    ADD CONSTRAINT "hardcopyMailingManifes2FK" FOREIGN KEY ("Administrator") REFERENCES "Administrator"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "paperAssignmentToolsRunBy"
    ADD CONSTRAINT "paperAssignmentToolsRu2FK" FOREIGN KEY ("Administrator") REFERENCES "Administrator"("ID") ON DELETE CASCADE;



ALTER TABLE ONLY "reviewerBiddingStartedBy"
    ADD CONSTRAINT "reviewerBiddingStarted2FK" FOREIGN KEY ("Administrator") REFERENCES "Administrator"("ID") ON DELETE CASCADE;
