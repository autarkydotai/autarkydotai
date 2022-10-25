#!/usr/bin/env python3

# Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
# 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022
# Python Software Foundation; All Rights Reserved
#
# Authors: Fred L. Drake, Jr. <fdrake@acm.org> (built-in CPython pprint module)
#          Nicolas Hug (scikit-learn specific changes)
#          Arnold Salas <asalas@autarky.ai> (autarkydotai specific changes)
#
# License: PSF License version 2 (see below)
#
# PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2
# --------------------------------------------
#
# 1. This LICENSE AGREEMENT is between the Python Software Foundation ("PSF"),
# and the Individual or Organization ("Licensee") accessing and otherwise
# using this software ("Python") in source or binary form and its associated
# documentation.
#
# 2. Subject to the terms and conditions of this License Agreement, PSF hereby
# grants Licensee a nonexclusive, royalty-free, world-wide license to
# reproduce, analyze, test, perform and/or display publicly, prepare
# derivative works, distribute, and otherwise use Python alone or in any
# derivative version, provided, however, that PSF's License Agreement and
# PSF's notice of copyright, i.e., "Copyright (c) 2001, 2002, 2003, 2004,
# 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016,
# 2017, 2018 Python Software Foundation; All Rights Reserved" are retained in
# Python alone or in any derivative version prepared by Licensee.
#
# 3. In the event Licensee prepares a derivative work that is based on or
# incorporates Python or any part thereof, and wants to make the derivative
# work available to others as provided herein, then Licensee hereby agrees to
# include in any such work a brief summary of the changes made to Python.
#
# 4. PSF is making Python available to Licensee on an "AS IS" basis. PSF MAKES
# NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR IMPLIED. BY WAY OF EXAMPLE, BUT
# NOT LIMITATION, PSF MAKES NO AND DISCLAIMS ANY REPRESENTATION OR WARRANTY OF
# MERCHANTABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF
# PYTHON WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.
#
# 5. PSF SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF PYTHON FOR ANY
# INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS AS A RESULT OF
# MODIFYING, DISTRIBUTING, OR OTHERWISE USING PYTHON, OR ANY DERIVATIVE
# THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.
#
# 6. This License Agreement will automatically terminate upon a material
# breach of its terms and conditions.
#
# 7. Nothing in this License Agreement shall be deemed to create any
# relationship of agency, partnership, or joint venture between PSF and
# Licensee. This License Agreement does not grant permission to use PSF
# trademarks or trade name in a trademark sense to endorse or promote products
# or services of Licensee, or any third party.
#
# 8. By copying, installing or otherwise using Python, Licensee agrees to be
# bound by the terms and conditions of this License Agreement.
#
#
# Brief summary of changes to original code (by Nicolas Hug):
# - "compact" parameter is supported for dicts, not just lists or tuples
# - estimators have a custom handler, they're not just treated as objects
# - long sequences (lists, tuples, dict items) with more than N elements are
#   shortened using ellipsis (', ...') at the end.
#
# Brief summary of changes to scikit-learn code (by Arnold Salas):
# - removed _changed_params() function and all references thereto
# - some refactoring, including lexicographic sorting of classes,
#   methods and functions
# - documentation and style edits.

"""This module contains our custom _PrettyPrinter class.

It is used in :meth:`~autarkydotai.utils.mixins.ReprMixin.__repr__` for
pretty-printing inherited class instance objects.

Classes
-------
KeyValTuple()
    Correctly render key-value tuples from dicts.
KeyValTupleParam()
    Correctly render key-value tuples from parameters.
_PrettyPrinter()
    Pretty printer class for class instance objects.

"""

import pprint
from collections import OrderedDict

from autarkydotai.utils import ReprMixin


class KeyValTuple(tuple):
    """Correctly render key-value tuples from dicts."""

    def __repr__(self):
        # Needed for _dispatch[tuple.__repr__] not to be overridden.
        return super().__repr__()


class KeyValTupleParam(KeyValTuple):
    """Correctly render key-value tuples from parameters."""

    pass


class _PrettyPrinter(pprint.PrettyPrinter):
    """Pretty printer class for class instance objects.

    This extends :class:`pprint.PrettyPrinter` because:
    - we want our own class instance objects to be displayed with their
      parameters, i.e. 'ClassName(param1=value1,...)', which is not
      supported by default.
    - the `compact` parameter in :meth:`pprint.PrettyPrinter.__init__`
      is ignored for dicts, which may lead to very long representations
      that we want to avoid.
    """

    def __init__(self, indent=1, width=80, depth=None, stream=None, *,
                 compact=False, indent_at_name=True,
                 n_max_elements_to_show=None):
        super().__init__(indent, width, depth, stream, compact=compact)
        self._indent_at_name = indent_at_name
        if self._indent_at_name:
            self._indent_per_level = 1  # Ignore indent param
        # Max number of elements in a list, dict, tuple until we start
        # using ellipses. This also affects the number of arguments of
        # a class instance object (they are treated as dicts).
        self.n_max_elements_to_show = n_max_elements_to_show

    def _format_dict_items(self, items, stream, indent, allowance, context,
                           level):
        return self._format_params_or_dict_items(
            items, stream, indent, allowance, context, level, is_dict=True
        )

    def _format_items(self, items, stream, indent, allowance, context, level):
        """Format the items of an iterable (list, tuple, etc.).

        Same as :meth:`pprint.PrettyPrinter._format_items`, but with
        additional support for ellipses if the number of elements to
        display is greater than ``self.n_max_elements_to_show``.
        """
        write = stream.write
        indent += self._indent_per_level
        if self._indent_per_level > 1:
            write((self._indent_per_level - 1) * ' ')
        delimnl = ',\n' + ' ' * indent
        delim = ''
        width = max_width = self._width - indent + 1
        it = iter(items)
        try:
            next_ent = next(it)
        except StopIteration:
            return
        last = False
        n_items = 0
        while not last:
            if n_items == self.n_max_elements_to_show:
                write(', ...')
                break
            n_items += 1
            ent = next_ent
            try:
                next_ent = next(it)
            except StopIteration:
                last = True
                max_width -= allowance
                width -= allowance
            if self._compact:
                rep = self._repr(ent, context, level)
                w = len(rep) + 2
                if width < w:
                    width = max_width
                    if delim:
                        delim = delimnl
                if width >= w:
                    width -= w
                    write(delim)
                    delim = ', '
                    write(rep)
                    continue
            write(delim)
            delim = delimnl
            self._format(ent, stream, indent, allowance if last else 1,
                         context, level)

    def _format_params_or_dict_items(self, object, stream, indent, allowance,
                                     context, level, is_dict=False):
        """Format dict items or parameters respecting ``compact=True``.

        Somehow, the built-in rendering of dict items does not respect
        ``compact=True`` and uses one line per key-value if it cannot
        fit everything in a single line. Thanks to this implementation,
        dict items will be rendered as '<key: value>', while parameters
        will be rendered as '<key=value>'.

        This custom implementation mostly involves copy/pasting code
        from :meth:`pprint.PrettyPrinter._format_items`. Additionally,
        it will insert ellipses if the number of items is greater than
        ``self.n_max_elements_to_show``.
        """
        write = stream.write
        indent += self._indent_per_level
        delimnl = ',\n' + ' ' * indent
        delim = ''
        width = max_width = self._width - indent + 1
        it = iter(object)
        try:
            next_ent = next(it)
        except StopIteration:
            return
        last = False
        n_items = 0
        while not last:
            if n_items == self.n_max_elements_to_show:
                write(', ...')
                break
            n_items += 1
            ent = next_ent
            try:
                next_ent = next(it)
            except StopIteration:
                last = True
                max_width -= allowance
                width -= allowance
            if self._compact:
                k, v = ent
                krepr = self._repr(k, context, level)
                vrepr = self._repr(v, context, level)
                if not is_dict:
                    krepr = krepr.strip("'")
                middle = ': ' if is_dict else '='
                rep = krepr + middle + vrepr
                w = len(rep) + 2
                if width < w:
                    width = max_width
                    if delim:
                        delim = delimnl
                if width >= w:
                    width -= w
                    write(delim)
                    delim = ', '
                    write(rep)
                    continue
            write(delim)
            delim = delimnl
            cls = KeyValTuple if is_dict else KeyValTupleParam
            self._format(cls(ent), stream, indent, allowance if last else 1,
                         context, level)

    def _pprint_key_val_tuple(self, object, stream, indent, allowance, context,
                              level):
        """Pretty printing for key-value tuples from dict or params."""
        k, v = object
        rep = self._repr(k, context, level)
        if isinstance(object, KeyValTupleParam):
            rep = rep.strip("'")
            middle = '='
        else:
            middle = ': '
        stream.write(rep)
        stream.write(middle)
        self._format(v, stream, indent + len(rep) + len(middle), allowance,
                     context, level)

    def _pprint_object(self, object, stream, indent, allowance, context,
                       level):
        stream.write(object.__class__.__name__ + '(')
        if self._indent_at_name:
            indent += len(object.__class__.__name__)
        params = object.get_params(deep=False)
        params = OrderedDict((k, v) for (k, v) in sorted(params.items()))
        self._format_params_or_dict_items(
            params.items(), stream, indent, allowance + 1, context, level
        )
        stream.write(')')

    def format(self, object, context, maxlevels, level):
        return _safe_repr(object, context, maxlevels, level)

    # Note: we need to copy _dispatch to prevent instances of the
    # pprint.PrettyPrinter() class from calling methods of our custom
    # _PrettyPrinter() class.
    # (see https://github.com/scikit-learn/scikit-learn/issues/12906)
    # mypy error: "Type[PrettyPrinter]" has no attribute "_dispatch"
    _dispatch = pprint.PrettyPrinter._dispatch.copy()  # type: ignore
    _dispatch[ReprMixin.__repr__] = _pprint_object
    _dispatch[KeyValTuple.__repr__] = _pprint_key_val_tuple


def _safe_repr(object, context, maxlevels, level):
    """Same as :func:`pprint._safe_repr`, but with added support for
    :class:`~autarkydotai.utils.mixins.ReprMixin` instances.
    """
    typ = type(object)
    if typ in pprint._builtin_scalars:
        return repr(object), True, False

    r = getattr(typ, '__repr__', None)
    if issubclass(typ, dict) and r is dict.__repr__:
        if not object:
            return '{}', True, False
        objid = id(object)
        if maxlevels and level >= maxlevels:
            return '{...}', False, objid in context
        if objid in context:
            return pprint._recursion(object), False, True
        context[objid] = 1
        readable = True
        recursive = False
        components = []
        append = components.append
        level += 1
        items = sorted(object.items(), key=pprint._safe_tuple)
        for k, v in items:
            krepr, kreadable, krecur = _safe_repr(k, context, maxlevels, level)
            vrepr, vreadable, vrecur = _safe_repr(v, context, maxlevels, level)
            append('%s: %s' % (krepr, vrepr))
            readable = readable and kreadable and vreadable
            if krecur or vrecur:
                recursive = True
        del context[objid]
        return '{%s}' % ', '.join(components), readable, recursive

    if (issubclass(typ, list) and r is list.__repr__) or \
         (issubclass(typ, tuple) and r is tuple.__repr__):
        if issubclass(typ, list):
            if not object:
                return '[]', True, False
            format = '[%s]'
        elif len(object) == 1:
            format = '(%s,)'
        else:
            if not object:
                return '()', True, False
            format = '(%s)'
        objid = id(object)
        if maxlevels and level >= maxlevels:
            return format % '...', False, objid in context
        if objid in context:
            return pprint._recursion(object), False, True
        context[objid] = 1
        readable = True
        recursive = False
        components = []
        append = components.append
        level += 1
        for o in object:
            orepr, oreadable, orecur = _safe_repr(o, context, maxlevels, level)
            append(orepr)
            if not oreadable:
                readable = False
            if orecur:
                recursive = True
        del context[objid]
        return format % ', '.join(components), readable, recursive

    # Added support for pretty-printing 'ReprMixin' subclasses.
    if issubclass(typ, ReprMixin):
        objid = id(object)
        if maxlevels and level >= maxlevels:
            return '{...}', False, objid in context
        if objid in context:
            return pprint._recursion(object), False, True
        context[objid] = 1
        readable = True
        recursive = False
        components = []
        append = components.append
        level += 1
        params = object.get_params(deep=False)
        items = sorted(params.items(), key=pprint._safe_tuple)
        for k, v in items:
            krepr, kreadable, krecur = _safe_repr(k, context, maxlevels, level)
            vrepr, vreadable, vrecur = _safe_repr(v, context, maxlevels, level)
            append('%s=%s' % (krepr.strip("'"), vrepr))
            readable = readable and kreadable and vreadable
            if krecur or vrecur:
                recursive = True
        del context[objid]
        return ('%s(%s)' % (typ.__name__, ', '.join(components)), readable,
                recursive)

    rep = repr(object)
    return rep, (rep and not rep.startswith('<')), False
