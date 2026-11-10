import unittest
from src.data.pilot import select_half
from src.evaluation.score import grade, evaluate


def question(**kwargs):
    return dict(id='T-1', grafico='BAR-001', tipo='Barras', estilo='cinza', nivel='N1',
                formato='número', correcao='±5%', resposta=-100, min=-105, max=-95) | kwargs


class EvaluationTests(unittest.TestCase):
    def test_limits_and_unparseable(self):
        for s in ['-105','−95,0']:
            self.assertTrue(grade(question(), s)[0])
        for s in ['-94,99','O valor é -100','nan','-100 ou 50','']:
            self.assertFalse(grade(question(), s)[0])

    def test_exact_and_order(self):
        q=question(correcao='exata', formato='lista', resposta='Grupo A, Grupo B')
        self.assertTrue(grade(q,'grupo a > grupo b')[0])
        self.assertFalse(grade(q,'Grupo B, Grupo A')[0])
        q=question(correcao='exata', formato='sim/não', resposta='Não')
        self.assertTrue(grade(q,'não.')[0])
        self.assertFalse(grade(q,'Sim e não')[0])

    def test_range(self):
        q=question(correcao='exata', formato='faixa', resposta='-20–-10')
        self.assertTrue(grade(q,'-20 a -10')[0])
        self.assertFalse(grade(q,'-20 a 10')[0])

    def test_unambiguous_thousands(self):
        q=question(resposta=1695,min=1610.25,max=1779.75)
        self.assertTrue(grade(q,'1.695,0')[0])
        self.assertFalse(grade(q,'1.695')[0])

    def test_missing_stays_in_denominator(self):
        s,_=evaluate([question(),question(id='T-2')], [{'id':'T-1','status':'ok','response':'-100'}])
        self.assertEqual(s['accuracy'], .5)
        self.assertEqual(s['missing_or_failed'],1)
        self.assertFalse(s['all_records_successful'])

    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([question()], [{'id':'T-1'},{'id':'T-1'}])

    def test_split_invariant_to_order(self):
        rows=[{'id':f'{t}-{n:03d}', 'style':s} for t in ['BAR','LIN']
              for s,indices in [('cinza',range(4)),('pastel',range(4,8))] for n in indices]
        a=select_half(rows)
        self.assertEqual(a,select_half(list(reversed(rows))))
        self.assertEqual(len(a),8)


if __name__=='__main__':
    unittest.main()
