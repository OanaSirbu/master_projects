negation(n(Y), Y):- !.
negation(Y, n(Y)).

concat_lists([], L2, L2).
concat_lists([X|L1], L2, [X|L3]):- concat_lists(L1, L2, L3).

elim_elem(_X, [], []).
elim_elem(X, [X|L], L):- !.
elim_elem(X, [Y|L], [Y|FinalList]):- elim_elem(X, L, FinalList).

check_res(A, L):- not(member(A, L)), not(is_taut(A)), not(is_subsumed(A, L)).

is_not_in_KB(_P, []).
is_not_in_KB(P, [L|KB]):- copy_term(P, P2), not(member(P2, L)), is_not_in_KB(P2, KB).
 
is_pure(L, KB):- member(P,L), negation(P, NP), copy_term(NP, NP2), is_not_in_KB(NP2, KB). 
 
elim_pure_clauses([], [],_).
elim_pure_clauses([L|KB], KB1,P):- append(KB,P,KB0),is_pure(L, KB0),!, elim_pure_clauses(KB, KB1,P).
elim_pure_clauses([L|KB], [L|KB1],P):- elim_pure_clauses(KB, KB1,[L|P]).
 
elim_pure_clauses(A,B):-elim_pure_clauses(A,B,[]).  

is_taut(L):- member(P, L), negation(P, NP), copy_term(NP, NP2), member(NP2, L), ground(P), ground(NP2).

elim_taut([], []).
elim_taut([L|KB], KB2):- is_taut(L), !, elim_taut(KB, KB2).
elim_taut([L|KB], [L|KB2]):- elim_taut(KB, KB2).

is_subsumed(L1, L2):- subset(L1, L2), L1 \== [].

elim_subsumed([], [], _).
elim_subsumed([L|KB], KB2, P):- append(KB, P, KB0), member(L2, KB0), copy_term(L2, SL),
    is_subsumed(SL, L), !, elim_subsumed(KB, KB2, P).
elim_subsumed([L|KB], [L|KB2], P):- elim_subsumed(KB, KB2, [L|P]).

elim_subsumed(KB, KB2):- elim_subsumed(KB, KB2, []).

final_res(BKB):- elim_pure_clauses(BKB, KB1), elim_taut(KB1, KB2), 
    elim_subsumed(KB2, KB3), res(KB3).

res(KB):- member([], KB), !, write('UNSAT'), nl.
res(KB):- 
    member(L1, KB), member(L2, KB), L1 \== L2,
    member(E1, L1), negation(E1, E2), copy_term(E2, X), member(X, L2),
    elim_elem(E1, L1, R1), elim_elem(X, L2, R2), 
    concat_lists(R1, R2, R3), sort(R3, R4), check_res(R4, KB), res([R4|KB]).
res(_):- write('SAT'), nl.

afis_lista([]):- nl.
afis_lista([X|A]):- write(X), tab(2), afis_lista(A).
%res1(KB,KB4):- elim_pure_clauses(KB, KB2), elim_taut(KB2, KB3), elim_subsumed(KB3, KB4).

read_kb(Stream):- at_end_of_stream(Stream).

read_kb(Stream):- not(at_end_of_stream(Stream)), read_line_to_codes(Stream, KB),
    read_term_from_codes(KB, L, []), afis_lista(L),
    final_res(L),
    read_kb(Stream).

main:- open('/home/oana/Documents/master-I/KRR/reasoning.txt', read, Stream),
    read_kb(Stream),
    close(Stream).


