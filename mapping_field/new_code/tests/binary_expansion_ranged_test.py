# TODO: do I need the following (old) test, or is it just enough to add a ranged promise to BinaryExpansion?
# def test_extend_range_to_full():
#     vv = [BoolVar(f'x_{i}') for i in range(4)]
#     x = BinaryExpansion(vv)
#
#     cond1 = (x < 16)
#     assert cond1 == TrueCondition
#
#     def from_mid(k: int):
#         cond1 = (x < k)
#         cond2 = (k <= x)
#         cond3 = RangeCondition(x, (k,16))
#         assert cond1 | cond2 == TrueCondition
#         assert cond2 | cond1 == TrueCondition
#         assert cond1 | cond3 == TrueCondition
#         assert cond3 | cond1 == TrueCondition
#
#     from_mid(15)
#     from_mid(9)
#     from_mid(8)
#     from_mid(1)
#     from_mid(0)

#
# def test_extend_range_partially():
#     vv = [BoolVar(f'x_{i}') for i in range(4)]
#     x = BinaryExpansion(vv)
#
#     def from_points(a: int, b: int, c: int):
#         cond1 = RangeCondition(x, (a,b))
#         cond2 = RangeCondition(x, (b,c))
#         result = RangeCondition(x, (a,c))
#         assert cond1 | cond2 == result
#         assert cond2 | cond1 == result
#
#     from_points(1,7,13)
#     from_points(1,8,13)
#     from_points(1,9,13)

#
# def test_extend_range_by_assignment():
#     vv = [BoolVar(f'x_{i}') for i in range(4)]
#     x = BinaryExpansion(vv)
#
#     cond1 = (x < 6)
#     for i in range(6, 19):
#         cond2 = x.as_assignment(i)
#         next_cond = (x < i+1)
#
#         union = cond1 | cond2
#         assert union == next_cond
#         union = cond2 | cond1
#         assert union == next_cond
#         cond1 = next_cond