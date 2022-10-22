# -*- coding: utf-8 -*-
"""
dd Discrimination Discovery module
@author: Salvatore Ruggieri
"""

import numpy as np
import pandas as pd
import pyroaring
import csv
import fim
import sys
import urllib
import gzip
import codecs
import heapq
import time
import math


def argmax(values, f):
    """ Argmax function

    Parameters:
    values (iterable): collection of values
    f (value->number): functional

    Returns:
    p: index of max value
    mv: max value, i.e., max{f(v) | v in values}
    """
    mv = None
    p = None
    for i, v in enumerate(values):
        fv = f(v)
        if mv is None or (fv is not None and mv < fv):
            mv, p = fv, i
    return p, mv


def getReader(filename, encoding='utf8'):
    """ Return a reader from a file, url, or gzipped file/url """
    if filename == '':
        return sys.stdin
    try:
        if filename.endswith('.gz'):
            file = gzip.open(filename)
        else:
            file = open(filename, encoding=encoding)
    except:
        file = urllib.request.urlopen(filename)
        if filename.endswith('.gz'):
            file = gzip.GzipFile(fileobj=file)
        reader = codecs.getreader(encoding)
        file = reader(file)
    return file


def getCSVattributes(freader, sep=','):
    """ Return the list of attributes in the header of a CSV or ARFF file reader """
    result = []
    line = freader.readline()
    while line == '':
        line = freader.readline()
    if line.startswith('@relation'):
        # read header from ARFF
        for line in freader:
            if line.startswith('@data'):
                break
            elif line.startswith('@attribute'):
                result.append(line.split(' ')[1])
    else:
        # read header from CSV
        result = line.strip().split(sep)
    return result


def get_att(itemDesc):
    """ Extract attribute name from attribute=value string """
    pos = itemDesc.find('=')
    return itemDesc[:pos] if pos >= 0 else ''


def get_val(itemDesc):
    """ Extract attribute name from attribute=value string """
    pos = itemDesc.find('=')
    return itemDesc[pos + 1:] if pos >= 0 else ''


def CSV2tranDB(filename, sep=',', na_values={'?'}, domains=dict(), codes=dict(), valuedecode=None):
    """ Read a CSV or ARFF file and encode it as a transaction database of attribute=value items

    Parameters:
    filename (string): filename, url, or gzipped file/url
    sep (string): column separator
    na_values (set: string coding missing values
    domains (dictionary): dictionary mapping binary variable names to values to be encoded. If a variable is not mapped, all values are encoded

    Returns:
    list: list of transactions
    dict: coding of items to numbers
    dict: decoding of numbers to items
    """
    with getReader(filename) as inputf:
        # header
        attributes = getCSVattributes(inputf, sep=sep)
        # reader for the rest of the file
        csvreader = csv.reader(inputf, delimiter=sep)
        nitems = max(codes.values()) + 1 if len(codes) > 0 else 0
        tDB = []
        # scan rows in CSV
        for values in csvreader:
            if len(values) == 0:
                continue
            # create transaction
            transaction = []
            for att, item in zip(attributes, values):
                if item in na_values:
                    continue
                if att in domains and item not in domains[att]:
                    continue
                attitem = att + "=" + (str(item) if valuedecode is None else valuedecode[att][item])
                code = codes.get(attitem)  # code of attitem
                if code is None:
                    codes[attitem] = code = nitems
                    nitems += 1
                transaction.append(code)
            # append transaction
            tDB.append(np.array(transaction, dtype=int))
    # decode list
    decodes = {code: attitem for attitem, code in codes.items()}
    return (tDB, codes, decodes)


def PD2tranDB(df, na_values={'NaN'}, domains=dict(), codes=dict(), valuedecode=None):
    """ Read a dataframe and encode it as a transaction database of attribute=value items

    Parameters:
    df (pd.DataFrame): dataframe
    na_values (set: string coding missing values
    domains (dictionary): dictionary mapping variable names to values to be encoded. If a variable is not mapped, all values are encoded
    codes (dictionary): extend existing codes
    decodes (dictionary): extend existing codes

    Returns:
    list: list of transactions
    dict: coding of items to numbers
    dict: decoding of numbers to items
    """
    nitems = max(codes.values()) + 1 if len(codes) > 0 else 0
    tDB = []
    for _, row in df.iterrows():
        transaction = []
        for att, item in row.iteritems():
            if item in na_values:
                continue
            if att in domains and item not in domains(att):
                continue
            attitem = att + "=" + (str(item) if valuedecode is None else valuedecode[att][item])
            code = codes.get(attitem)  # code of attitem
            if code is None:
                codes[attitem] = code = nitems
                nitems += 1
            transaction.append(code)
        # append transaction
        tDB.append(np.array(transaction, dtype=int))
    # decode list
    decodes = {code: attitem for attitem, code in codes.items()}
    return (tDB, codes, decodes)


class Encode:
    """ Encoding of discrete attributes in dataframes """

    def __init__(self, atts=None, decode=dict()):
        """ Constructor

        Parameters:
        atts (iterable): attributes to encode or None for all attributes
        decode (dictionary): pre-set encodings to reuse
        """
        self.atts = atts
        self.decode = decode
        self.encode = {c: {i: v for v, i in self.decode[c].items()} for c in self.decode}

    def fit_transform(self, df):
        """ Encode a dataframe of categories

        Parameters:
        df (pd.DataFrame): dataframe

        Returns:
        pd.DataFrame: encoded dataframe
        """
        res = pd.DataFrame()
        self.atts = df.columns if self.atts is None else self.atts
        self.encode = {c: {i: v for v, i in self.decode[c].items()} for c in self.decode}
        for col in self.atts:
            if col not in self.encode:
                uniq = sorted([v for v in df[col].unique() if pd.notna(v)])
                self.encode[col] = {v: i for i, v in enumerate(uniq)}
            res[col] = df[col].map(self.encode[col])
            # res.loc[df[col].isnull(), col] = np.nan
            res[col] = res[col].astype('category')
        self.decode = {c: {i: v for v, i in self.encode[c].items()} for c in self.encode}
        for col in set(df.columns) - set(self.atts):
            res[col] = df[col]
        return res


class tDBIndex:
    """ A transaction database index storing covers of item in bitmaps """

    def __init__(self, tDB, items=None):
        """ Create database index from a list of transactions """
        if items is None:
            items = {item for t in tDB for item in t}
        covers = {item: pyroaring.BitMap() for item in items}
        for tid, t in enumerate(tDB):
            for item in t:
                covers[item].add(tid)
        self.covers = {s: pyroaring.FrozenBitMap(b) for s, b in covers.items()}
        self.ncolumns = len(items)
        self.nrows = len(tDB)

    def cover(self, itemset, base=None):
        """ Return cover of an itemset (list of items) """
        nitems = len(itemset)
        if nitems == 0:
            return pyroaring.BitMap(np.arange(self.nrows)) if base is None else base
        if base is None:
            return pyroaring.BitMap.intersection(*[self.covers[item] for item in itemset])
        return pyroaring.BitMap.intersection(base, *[self.covers[item] for item in itemset])

    def cover_none(self):
        """ Return empty cover """
        return pyroaring.BitMap([])

    def cover_all(self):
        """ Return cover all set """
        return pyroaring.BitMap(np.arange(self.nrows))

    def supp(self, itemset, base=None):
        """ Return support of an itemset (list of items) """
        return len(self.cover(itemset, base))


class ContingencyTable:
    """ A contingency table.

     contingency table for inference (unprotected=true.good, n1=a,n2=c)
     ===========
     true.bad      a
     true.good     c
     ===========   n()

     confusion matrix (unprotected=true.good)
     =========== pred.bad === pred.good ===
     true.bad        a            b       n1()
     true.good       c            d       n2()
     ===========    m1()  ===    m2()  ==  n()

     contingency table for independence
     =========== pred.bad === pred.good ===
     protected       a            b       n1()
     unprotected     c            d       n2()
     ===========    m1()  ===    m2()  ==  n()

     contingency table for separation
          protected                                   unprotected
     ========= pred.bad  ==  pred.good  ===   ====  pred.bad  ==  pred.good  ===
     true.bad    TPp          FNp      Pp()           TPu           FNu      Pu()
     true.good   FPp          TNp      Np()           FPu           TNu      Nu()
     ==========   a     =====  b  ===  n1()   ====     c    ====     d  ===  n2()

    """

    def __init__(self, a, n1, c, n2, TPp=None, Pp=None, TPu=None, Pu=None,
                 ctx=None, ctx_n=0, protected=None):
        """ Init the contingency table. """
        self.ctx = ctx
        self.ctx_n = int(ctx_n)
        self.protected = protected
        # store both contingency table for independence
        self.a = int(a)
        self.b = int(n1 - a)
        self.c = int(c)
        self.d = int(n2 - c)
        # and for separation
        nNone = int(TPp is None) + int(Pp is None) + int(TPu is None) + int(Pu is None)
        if nNone == 0:
            self.TPp = int(TPp)
            self.FPp = self.a - self.TPp
            self.FNp = int(Pp) - self.TPp
            self.TNp = self.b - self.FNp
            self.TPu = int(TPu)
            self.FPu = self.c - self.TPu
            self.FNu = int(Pu) - self.TPu
            self.TNu = self.d - self.FNu
        elif nNone != 4:
            raise "TPp, Pp, TPu, Pu must all be given"

    def __lt__(self, other):
        """ lexicographic ordering on ctx """
        return self.ctx < other.ctx

    def __eq__(self, other):
        """ lexicographic ordering on ctx """
        return self.ctx == other.ctx

    def __hash__(self):
        """ hash on on ctx """
        return hash(self.ctx)

    def n1(self):
        """ a+b in contingency table """
        return self.a + self.b

    def n2(self):
        """ c+d in contingency table """
        return self.c + self.d

    def n(self):
        """ a+b+c+d in contingency table """
        return self.a + self.b + self.c + self.d

    def m1(self):
        """ a+c in contingency table """
        return self.a + self.c

    def m2(self):
        """ b+d in contingency table """
        return self.b + self.d

    def acc(self):
        """ predictive accuracy """
        return (self.a + self.d) / self.n()

    def err(self):
        """ predictive accuracy """
        return (self.b + self.c) / self.n()

    def p1(self):
        """ a/n1, i.e., precision of bad """
        return self.a / self.n1()

    def p2(self):
        """ c/n2, i.e., precision of good """
        return self.c / self.n2()

    def p(self):
        """ m1/n, i.e., proportion of bad """
        return self.m1() / self.n()

    def rd(self):
        """ p1-p2, i.e., risk difference """
        return self.p1() - self.p2()

    def ed(self):
        """ p1-p, i.e., extended risk difference """
        return self.p1() - self.p()

    def rr(self):
        """ p1/p2, i.e., risk ratio """
        p2 = self.p2() if self.c > 0 else 1 / (self.d + 1)
        return self.p1() / p2

    def er(self):
        """ p1/p, i.e., extended risk ratio """
        return self.p1() / self.p()

    def rc(self):
        """ (1-p1)/(1-p2), i.e., risk chance """
        oneminusp2 = (1 - self.p2()) if self.d > 0 else 1 / (self.c + 1)
        return (1 - self.p1()) / oneminusp2

    def ec(self):
        """ (1-p1)/(1-p), i.e., extended risk chance """
        return (1 - self.p1()) / (1 - self.p())

    def orisk(self):
        """ p1/(1-p1)*(1-p2)/p2, i.e., odds risk """
        p1 = self.p1();
        p2 = self.p2()
        return p1 / (1 - p1) * (1 - p2) / p2

    def Pp(self):
        """ TPp+FNp """
        return self.TPp + self.FNp

    def Np(self):
        """ FPp+TNp """
        return self.FPp + self.TNp

    def Pu(self):
        """ TPu+FNu """
        return self.TPu + self.FNu

    def Nu(self):
        """ FPu+TNu """
        return self.FPu + self.TNu

    def accu(self):
        """ predictive accuracy unprotected group """
        return (self.TPu + self.TNu) / self.n2()

    def accp(self):
        """ predictive accuracy protected group """
        return (self.TPp + self.TNp) / self.n1()

    def tpru(self):
        """ TPR unprotected group """
        return self.TPu / self.Pu()

    def tprp(self):
        """ TPR protected group """
        return self.TPp / self.Pp()

    def tnru(self):
        """ TNR unprotected group """
        return self.TNu / self.Nu()

    def tnrp(self):
        """ TNR protected group """
        return self.TNp / self.Np()

    def eop(self):
        """ EOP equality of opportunity """
        return self.tnru() - self.tnrp()


class DD:
    """ Discrimination discovery class. """

    def __init__(self, df, unprotectedItem, predBadItem=None, trueBadItem=None,
                 na_values={'NaN', 'nan', '?'}, domains=dict(),
                 codes=dict(), valuedecode=None):
        """ Init with given parameters

        Parameters:
        df (pd.DataFrame): dataframe
        unprotectedItem (string): item of unprotected group in the form "att_name=value", e.g., "sex=male". All other values of att_name will be considered as protected grounds
        predBadItem (string): item of negative decision in the form "att_name=value", e.g., "grant=deny". The att_name is assumed to be binary
        trueBadItem (string): optional item of correct negative decision in the form "att_name=value", e.g., "will_repay=no". The att_name is assumed to be binary
        na_values (set: string coding missing values
        domains (dictionary): dictionary mapping binary variable names to values to be encoded. If a variable is not mapped, all values are encoded
        """
        if isinstance(df, pd.DataFrame):
            self.tDB, self.codes, self.decodes = PD2tranDB(df, na_values=na_values, domains=domains, codes=codes,
                                                           valuedecode=valuedecode)
        else:
            self.tDB, self.codes, self.decodes = CSV2tranDB(df, na_values=na_values, domains=domains, codes=codes,
                                                            valuedecode=valuedecode)
        self.decodes[-1] = unprotectedItem.replace('=', '!=')
        # print(self.decodes)
        self.itDB = tDBIndex(self.tDB, self.codes.values())

        self.unprotectedItem = unprotectedItem
        self.sensitiveAtt = get_att(unprotectedItem)
        self.unprotected = self.codes[unprotectedItem]
        self.unprCover = self.itDB.covers[self.unprotected]
        self.protected = [self.codes[v] for v in self.codes if
                          get_att(v) == self.sensitiveAtt and self.codes[v] != self.unprotected]

        self.predBadItem = predBadItem
        if predBadItem is None:
            self.predAtt = self.predBad = self.predGood = self.predGoodItem = None
            self.predBadCover = self.predBadAVG = None
            trueBadItem = None  # force to None
        else:
            self.predAtt = get_att(predBadItem)
            self.predBad = self.codes[predBadItem]
            predGoods = [self.codes[v] for v in self.codes if
                         get_att(v) == self.predAtt and self.codes[v] != self.predBad]
            assert len(predGoods) == 1, "binary decisions only!"
            self.predGood = predGoods[0]
            self.predGoodItem = self.decodes[self.predGood]
            self.predBadCover = self.itDB.covers[self.predBad]
            self.predBadAVG = len(self.predBadCover) / self.itDB.nrows

        self.trueBadItem = trueBadItem
        if trueBadItem is None:
            self.trueAtt = self.trueBad = self.trueGood = self.trueGoodItem = None
            self.trueBadAVG = self.unprTrueBadCover = None
        else:
            self.trueAtt = get_att(trueBadItem)
            self.trueBad = self.codes[trueBadItem]
            posTrue_s = [self.codes[v] for v in self.codes if
                         get_att(v) == self.trueAtt and self.codes[v] != self.trueBad]
            assert len(posTrue_s) == 1, "binary decisions only!"
            self.trueGood = posTrue_s[0]
            self.trueGoodItem = self.decodes[self.trueGood]
            self.trueBadCover = self.itDB.covers[self.trueBad]
            self.trueBadAVG = len(self.trueBadCover) / self.itDB.nrows
            self.unprTrueBadCover = self.unprCover & self.trueBadCover

    def ctg(self, itemset, protected, cover=None):
        if len(itemset) > 0 and isinstance(itemset[0], str):
            itemset = [self.codes[i] for i in itemset]
        ctx = self.itDB.cover(itemset) if cover is None else cover
        if self.predBadItem is None:
            c = ctx.intersection_cardinality(self.unprCover)
            if protected == -1:  # any protected
                prCover = self.itDB.cover_none()
                for pr in self.protected:
                    prCover |= self.itDB.covers[pr]
            else:
                prCover = self.itDB.covers[protected]
            a = ctx.intersection_cardinality(prCover)
            ctg = ContingencyTable(a=a, n1=a, c=c, n2=c, TPp=None, Pp=None,
                                   TPu=None, Pu=None, ctx=itemset, ctx_n=len(ctx), protected=protected)
        else:
            n2 = ctx.intersection_cardinality(self.unprCover)
            ctxPredBadCover = ctx & self.predBadCover
            c = ctxPredBadCover.intersection_cardinality(self.unprCover)
            if self.trueBad is None:
                TPu = Pu = None
            else:
                TPu = ctxPredBadCover.intersection_cardinality(self.unprTrueBadCover)
                Pu = ctx.intersection_cardinality(self.unprTrueBadCover)
            if protected == -1:  # any protected
                prCover = self.itDB.cover_none()
                for pr in self.protected:
                    prCover |= self.itDB.covers[pr]
            else:
                prCover = self.itDB.covers[protected]
            n1 = ctx.intersection_cardinality(prCover)
            a = ctxPredBadCover.intersection_cardinality(prCover)
            if self.trueBad is None:
                TPp = Pp = None
            else:
                prTrueBadCover = prCover & self.trueBadCover
                TPp = ctxPredBadCover.intersection_cardinality(prTrueBadCover)
                Pp = ctx.intersection_cardinality(prTrueBadCover)
            ctg = ContingencyTable(a=a, n1=n1, c=c, n2=n2, TPp=TPp, Pp=Pp,
                                   TPu=TPu, Pu=Pu, ctx=itemset, ctx_n=len(ctx), protected=protected)
        return ctg

    def ctg_global(self, itemset=[]):
        """ Return contingency table(s) for the whole dataset """
        return [self.ctg(itemset, protected) for protected in self.protected]

    def ctg_rel(self, ctg, base=None):
        """ Return contingency table by changing the context to a given bitmap """
        if base is None:
            return self.ctg(ctg.ctx, ctg.protected, cover=self.itDB.cover(ctg.ctx))
        return self.ctg(ctg.ctx, ctg.protected, cover=self.itDB.cover(ctg.ctx) & base)

    def ctg_any(self, itemset=[], cover=None):
        """ Return contingency table(s) for a specified coverage and ANY protected """
        return self.ctg(itemset, -1, cover=cover)

    def extract(self, minSupp=20, testCond=lambda x: 0, topk=0, target='c'):
        """ Extract top-k contingency tables with minimum support and satisfying a test condition

        Parameters:
        minSupp (int): minimum support of contingency table context (negative = absolute, positive = percentage)
        testCond (functional): a function testing a contingency table. testCond(ct) will return None if ct is not to be considered, and a numeric value to be used in ordering contingency tables otherwise
        topk (int): maximum number of contingency tables in output. The top-k will be outputed wrt the testCond() output
        target (string): type of frequent itemsets ('c' for closed, 's' frequent, 'm' maximal)

        Returns:
        list: list of pairs (testCond(ct), ct) where ct is a topk contingency table
        """
        exclude = {self.codes[v] for v in self.codes if get_att(v) in {self.sensitiveAtt, self.predAtt, self.trueAtt}}
        tDBprojected = [list(set(t) - exclude) for t in self.tDB]
        fisets = fim.fpgrowth(tDBprojected, supp=minSupp, zmin=0, target=target)
        q = []
        if self.predBadItem is None:
            for fi in fisets:
                ctx = self.itDB.cover(fi[0])
                c = ctx.intersection_cardinality(self.unprCover)
                for protected in self.protected:
                    prCover = self.itDB.covers[protected]
                    a = ctx.intersection_cardinality(prCover)
                    ctg = ContingencyTable(a=a, n1=a, c=c, n2=c, TPp=None, Pp=None,
                                           TPu=None, Pu=None, ctx=fi[0], ctx_n=int(fi[1]), protected=protected)
                    v = testCond(ctg)
                    if v is not None and v != False:
                        if len(q) < topk:
                            heapq.heappush(q, (v, ctg))
                        else:
                            heapq.heappushpop(q, (v, ctg))
        else:
            for fi in fisets:
                ctx = self.itDB.cover(fi[0])
                n2 = ctx.intersection_cardinality(self.unprCover)
                ctxPredBadCover = ctx & self.predBadCover
                c = ctxPredBadCover.intersection_cardinality(self.unprCover)
                if self.trueBad is None:
                    TPu = Pu = None
                else:
                    TPu = ctxPredBadCover.intersection_cardinality(self.unprTrueBadCover)
                    Pu = ctx.intersection_cardinality(self.unprTrueBadCover)
                for protected in self.protected:
                    prCover = self.itDB.covers[protected]
                    n1 = ctx.intersection_cardinality(prCover)
                    a = ctxPredBadCover.intersection_cardinality(prCover)
                    if self.trueBad is None:
                        TPp = Pp = None
                    else:
                        prTrueBadCover = prCover & self.trueBadCover
                        TPp = ctxPredBadCover.intersection_cardinality(prTrueBadCover)
                        Pp = ctx.intersection_cardinality(prTrueBadCover)
                    ctg = ContingencyTable(a=a, n1=n1, c=c, n2=n2, TPp=TPp, Pp=Pp,
                                           TPu=TPu, Pu=Pu, ctx=fi[0], ctx_n=int(fi[1]), protected=protected)
                    v = testCond(ctg)
                    if v is not None and v != False:
                        if len(q) < topk:
                            heapq.heappush(q, (v, ctg))
                        else:
                            heapq.heappushpop(q, (v, ctg))
        if len(self.protected) > 1:
            ms = -minSupp if minSupp < 0 else int(minSupp * self.itDB.nrows)
            q = [ctg for ctg in q if ctg[1].n() >= ms]
        return sorted(q, reverse=True)

    def ctg_info(self, ctg):
        """ Return context and protected value """
        protectedDesc = ' ' if self.unprotected is None else self.decodes[ctg.protected]
        if ctg.ctx != [-1]:
            ctx = ' AND '.join([self.decodes[it] for it in ctg.ctx]) if ctg.ctx != [] else 'ALL'
            return ctx, protectedDesc
        return "<extensional>", protectedDesc

    def print(self, ctg):
        """ Pretty print of a contingency table ctg """
        ctxDesc, protectedDesc = self.ctg_info(ctg)
        n = ctg.n()
        print('-----\nContext', ctxDesc)
        print('Size = {}  Perc = {:.2f}%'.format(ctg.ctx_n, 100.0 * ctg.ctx_n / self.itDB.nrows))
        if self.predBadItem is None:
            xlen = max(len(protectedDesc), len(self.unprotectedItem))
            spec = ('{:' + str(xlen) + '}|{:' + str(len(str(n))) + '}')
            print(spec.format('', ''))
            print(spec.format(protectedDesc, ctg.a))
            print(spec.format(self.unprotectedItem, ctg.c))
            print(spec.format('', ctg.m1()))
        elif self.trueBad is None:
            xlen = max(len(protectedDesc), len(self.unprotectedItem))
            spec = ('{:' + str(xlen) + '}|{:' + str(len(self.predBadItem)) + '}|{:' + str(
                len(self.predGoodItem)) + '}|{:' + str(len(str(n))) + '}')
            print(spec.format('', self.predBadItem, self.predGoodItem, ''))
            print(spec.format(protectedDesc, ctg.a, ctg.b, ctg.n1()))
            print(spec.format(self.unprotectedItem, ctg.c, ctg.d, ctg.n2()))
            print(spec.format('', ctg.m1(), ctg.m2(), n))
        else:
            xlen = max(len(self.trueBadItem), len(self.trueGoodItem))
            spec = '{:' + str(
                xlen + 3 + len(self.predBadItem) + len(self.predGoodItem) + len(str(ctg.n1()))) + '}      ' + '{}'
            print(spec.format(protectedDesc, self.unprotectedItem))
            spec = ('{:' + str(xlen) + '}|{:' + str(len(self.predBadItem)) + '}|{:' + str(
                len(self.predGoodItem)) + '}|{:' + str(len(str(n))) + '}')
            spec = spec + '     ' + spec
            print(spec.format('', self.predBadItem, self.predGoodItem, '', '', self.predBadItem, self.predGoodItem, ''))
            print(
                spec.format(self.trueBadItem, ctg.TPp, ctg.FNp, ctg.Pp(), self.trueBadItem, ctg.TPu, ctg.FNu, ctg.Pu()))
            print(spec.format(self.trueGoodItem, ctg.FPp, ctg.TNp, ctg.Np(), self.trueGoodItem, ctg.FPu, ctg.TNu,
                              ctg.Nu()))
            print(spec.format('', ctg.a, ctg.b, ctg.n1(), '', ctg.c, ctg.d, ctg.n2()))

    def cover_n(self, patterns, f, k=None, only_protected=True):
        """ Naive max cover

        Parameters:
        patterns (iterable): collection of patterns (contingency tables)
        f (pattern, bitmap->number): pattern importance relative to uncovered subset
        k (int): max number of patterns in cover

        Returns:
        list: greedy cover of argmax_{x subseteq aset, |x| <= k} f(x)
        """
        db = self.itDB
        start_time = time.perf_counter()
        pset = list(patterns)
        covers = []
        residuals = []
        times = []
        active = db.cover_all()  # population under consideration
        n_pr_uncovered = db.nrows - len(self.unprCover)  # no protected uncovered
        if k is None or k >= len(pset):
            k = len(pset)
        while k > 0 and n_pr_uncovered > 0:
            bestp, bestv = argmax(pset, lambda ctg: f(self.ctg_rel(ctg, base=active)))
            if bestv is None:
                break
            bestb = self.ctg_rel(pset[bestp], base=active)
            if only_protected:
                new_cov = db.cover(bestb.ctx + (bestb.protected,), base=active)
            else:
                new_cov = db.cover(bestb.ctx, base=active)
            new_supp = len(new_cov)  # newly covered protected
            if new_supp == 0:
                break
            tm = time.perf_counter() - start_time
            covers.append(bestb)
            residuals.append(new_supp)
            times.append(tm)
            active -= new_cov
            n_pr_uncovered -= new_supp
            # print(bestv, new_supp, nuncovered)
            k -= 1
            del pset[bestp]
        all_covered = db.cover_none()
        for c in covers:
            all_covered |= db.cover(c.ctx)
        ctg_cov = self.ctg_any([-1], cover=all_covered)
        ctg_uncov = self.ctg_any([-1], cover=db.cover_all() - all_covered)
        return (covers, residuals, times, active - self.unprCover, ctg_cov, ctg_uncov)


class ID:
    """ Distance functions and individual discrimination utils. """

    def __init__(self, df, nominal_atts=[], continuous_atts=[], ordinal_atts=[]):
        """ Init with given parameters

        Parameters:
        df (pd.DataFrame): dataframe, used to collect statistics about distributions
        nominal_atts (iterable): nominal feature names
        continuous_atts (iterable): continuous feature names
        ordinal_atts (iterable): ordinal feature names
        """
        self.nominal_atts = nominal_atts
        self.continuous_atts = continuous_atts
        self.ordinal_atts = ordinal_atts
        self.natts = len(continuous_atts) + len(nominal_atts) + len(ordinal_atts)
        # statistics of continuous features
        self.means = {c: df[c].mean() for c in continuous_atts}
        self.stds = {c: df[c].std() for c in continuous_atts}
        # statistics of ordinal feastures
        self.nofvalues = {c: (df[c].nunique() - 1) for c in ordinal_atts}
        # positions of features (for future usage)
        cols = list(df.columns)
        self.nominal_atts_pos = [cols.index(c) for c in nominal_atts]
        self.continuous_atts_pos = [cols.index(c) for c in continuous_atts]
        self.ordinal_atts_pos = [cols.index(c) for c in ordinal_atts]
        self.stds_pos = {self.continuous_atts_pos[i]: self.stds[c] for i, c in enumerate(continuous_atts)}
        self.nofvalues_pos = {self.ordinal_atts_pos[i]: self.nofvalues[c] for i, c in enumerate(ordinal_atts)}

    def kdd2011dist(self, t, tset):
        """ Distance function used in the KDD 2011 paper.

        Parameters:
        t (dictionary): instance to compute distance from
        tset (DataFRame or dictionary): set of instances to compute distance to

        Returns:
        pd.Series or float: distances
        """
        if isinstance(tset, pd.DataFrame):
            tot = pd.Series(np.zeros(len(tset)), index=tset.index)
            for c in self.continuous_atts:
                dist = abs(t[c] - tset[c]) / self.stds[c]
                dist[dist.isnull()] = 3
                tot += dist
            for c in self.nominal_atts:
                tot += 1 * (t[c] != tset[c])  # notice t[c]!=tset[c] is True if one or both are NaN
            for c in self.ordinal_atts:
                nvals = self.nofvalues[c]
                val = t[c] / nvals
                tmp = tset[c] / nvals
                if math.isnan(val):
                    dist = max(tmp, 1 - tmp)
                    dist[dist.isnull()] = 1
                else:
                    dist = abs(val - tmp)
                    dist[dist.isnull()] = max(val, 1 - val)
                tot += dist
        else:
            tot = 0
            for c in self.continuous_atts:
                dist = 3 if math.isnan(tset[c]) else abs(t[c] - tset[c]) / self.stds[c]
                tot += dist
            for c in self.nominal_atts:
                tot += 1 * (t[c] != tset[c])  # notice t[c]!=tset[c] is True if one or both are NaN
            for c in self.ordinal_atts:
                nvals = self.nofvalues[c]
                val = t[c] / nvals
                tmp = tset[c] / nvals
                if math.isnan(val):
                    dist = 1 if math.isnan(tmp) else max(tmp, 1 - tmp)
                else:
                    dist = max(val, 1 - val) if math.isnan(tmp) else abs(val - tmp)
                tot += dist
        return tot / self.natts

    def topk(self, t, tset, distf, k, maxd=None):
        """ Compute top-k instances close to a given one.

        Parameters:
        t (dictionary): instance to compute distance from
        tset (DataFrame or dictionary): set of instances to compute distance to
        distf (function): distance function
        k (int): number of closest instances
        maxd (float): max distance between t and the closest ones

        Returns:
        list of pairs: list of (distance, index) of the up-to-k closest instances to t that are at a distance of at most maxd
        """
        ds = distf(t, tset)  # todo: why not just use kdd?
        q = []
        lenq = 0
        for i, d in zip(tset.index, ds):
            if maxd is None or d <= maxd:
                if lenq < k:
                    heapq.heappush(q, (-d, i))
                    lenq += 1
                else:
                    d1, _ = heapq.heappushpop(q, (-d, i))
                    maxd = -d1
        q = [(-v, i) for v, i in q]
        return sorted(q)

    def topkdiff(self, df, unpro, pro, predBadItem, distf, k, maxd=None):
        """ Compute risk difference for each instance in protected set.

        Parameters:
        df (DataFrame): dataset
        unpro (Series): condition for unprotected instances
        pro (Series or list): condition for protected instances, or list of conditions for a number of protected groups
        class_att (string): decision attribute
        distf (function): distance function
        k (int): number of closest instances
        maxd (float): max distance between t and the closest ones

        Returns:
        Series: risk difference between topk protected and topk unprotected for protected instances, and zero for all other instances in the dataframe df
        """
        class_att = get_att(predBadItem)
        bad_dec = int(get_val(predBadItem))
        res = pd.Series(np.zeros(len(df)), index=df.index)
        unpro_set = df[unpro]
        if not isinstance(pro, list):
            pro = [pro]
        for protected in pro:
            pro_set = df[protected]
            for i, rowp in pro_set.iterrows():
                res1 = self.topk(rowp, pro_set, distf, k + 1, maxd=maxd)
                res2 = self.topk(rowp, unpro_set, distf, k, maxd=maxd)
                nn1 = [j for _, j in res1 if j != i]
                nn2 = [j for _, j in res2]
                # efficient but specific of RD
                p1 = sum(pro_set.loc[nn1, class_att] == bad_dec) / len(nn1)
                p2 = sum(unpro_set.loc[nn2, class_att] == bad_dec) / len(nn2)
                res.loc[i] = p1 - p2
        return res


""" Sample usage of DD"""
if __name__ == '__main__':

    def check_acc(ctg):
        n = ctg.n()
        if n == 0:
            return None  # this may occur for relative contingency table
        acc = max(ctg.a, ctg.c) / n
        return (int(acc * n / 10), acc) if acc > 0.9 and n > 10 else None


    start_time = time.perf_counter()

    # disc = DD("../data/credit.csv", 'age=from_41d4_le_52d6', 'class=bad')
    disc = DD("../data/credit.csv", 'class=good')
    print('== Global ==')
    for ctg in disc.ctg_global():
        disc.print(ctg)
        # print("RD = {:f}".format(ctg.rd()))
        # print("ACC = {:.2f}".format(check_acc(ctg)))
    ctgs = disc.extract(testCond=check_acc, minSupp=-20, topk=2000)
    print('== Top ==')
    for v, ctg in ctgs[:2]:
        disc.print(ctg)
        # print("RD = {:f}".format(v))
        # print("ACC = {:.2f}".format(check_acc(ctg)))
    print('== Rel top1 ==')
    disc.print(disc.ctg_rel(ctgs[0][1], disc.itDB.cover_all()))
    covers, residuals, times, uncovered, ctg_cov, ctg_uncov = disc.cover_n([ctg for _, ctg in ctgs], check_acc, 100)
    print('== Cover ==')
    for ctg, res in zip(covers, residuals):
        print(res)
        disc.print(ctg)
        # print("ACC = {:.2f}".format(check_acc(ctg)))
        # print("RD = {:f}".format(ctg.rd()))

    elapsed_time = time.perf_counter() - start_time
    print('Elapsed time (s): {:.2f}'.format(elapsed_time))
    print('Contingency tables: {}'.format(len(ctgs)))

#
# EOF @ 16/09/22
#
