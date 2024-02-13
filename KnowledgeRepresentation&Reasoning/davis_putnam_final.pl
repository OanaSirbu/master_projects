negation(n(Y), Y):- !.
negation(Y, n(Y)).

no_occur(_X, [], 0).
no_occur(X, [X|T], Nr):- no_occur(X, T, Nr2), Nr is Nr2 + 1.
no_occur(X, [Y|T], Nr):- X \== Y, no_occur(X, T, Nr).

lit_with_most_occur(L, Lit):- flatten(L, OneL), 
    setof(Nr-Lit, (member(Lit, OneL),no_occur(Lit, OneL, Nr)), LitList),
    max_member(_MaxNr-Lit, LitList).

find_balance(Lit, L, Balance):- negation(Lit, NLit),
    no_occur(Lit, L, PosNoLit),
    no_occur(NLit, L, NegNoLit),
    Balance is abs(PosNoLit - NegNoLit).

lit_most_balanced(KB, ChosenLit):-
    flatten(KB, OneL),
    setof(Balance-Lit, (member(Lit, OneL), find_balance(Lit, OneL, Balance)), Balances),
    keysort(Balances, SortedBalances),
    SortedBalances = [_MinBalance-ChosenLit|_].

%check(OneL, LitList):- setof(Nr-Lit, (member(Lit, OneL),no_occur(Lit, OneL, Nr)), LitList).

is_empty(L):- length(L, 0).


doesnt_contain_p_and_np(P, L):- not(member(P, L)), negation(P, NP), not(member(NP, L)).
first_dot(L, P, R):- include(doesnt_contain_p_and_np(P), L, R).


without_p(P, L):- not(member(P, L)).
with_np(NP, L):- member(NP, L).

elim_np_from_clause(C, NP2, Res1):- delete(C, NP2, Res1). 

elim_np_from_clauses([], _P, []).
elim_np_from_clauses([C|L], P, [Res1|Res2]):- negation(P, NP),
    elim_np_from_clause(C, NP, Res1), elim_np_from_clauses(L, P, Res2).
%elim_np_from_clauses([C|L], P, [C|Res2]):- elim_np_from_clauses(L, P, Res2).

second_dot(L, P, R1):- negation(P, NP), 
    include(without_p(P), L, R), include(with_np(NP), R, R2),
    elim_np_from_clauses(R2, P, R1).


dot_op(C, P, FinalResult):- first_dot(C, P, R),second_dot(C, P, R1), append(R, R1, FinalResult).


dp1([],[]):- write('yes'), nl.
dp1(C,_):- member([], C), write('no'), nl, !.
dp1(C,[P/true|S]):- lit_with_most_occur(C, P), dot_op(C, P, L1), dp1(L1,S),!.
dp1(C,[P/false|S]):- lit_with_most_occur(C, P), negation(P, NP), dot_op(C, NP, L2),dp1(L2, S).

dp2([],[]):- write('yes'), nl.
dp2(C,_):- member([], C), write('no'), nl, !.
dp2(C,[P/true|S]):- lit_most_balanced(C, P), dot_op(C, P, L1), dp2(L1,S),!.
dp2(C,[P/false|S]):- lit_most_balanced(C, P), negation(P, NP), dot_op(C, NP, L2),dp2(L2, S).

afis_lista([]):- nl.
afis_lista([X|A]):- write(X), tab(2), afis_lista(A).

read_kb(Stream):- at_end_of_stream(Stream).

read_kb(Stream):- not(at_end_of_stream(Stream)),
    read_line_to_codes(Stream, KB),
    read_term_from_codes(KB, L, []), afis_lista(L),
    dp1(L, S1), afis_lista(S1), 
    dp2(L, S2), afis_lista(S2), nl,
    read_kb(Stream).

main:- open('/home/oana/Documents/master-I/KRR/dp.txt', read, Stream),
    read_kb(Stream),
    close(Stream).
