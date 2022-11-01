# Copyright (c) 2007-2022 The scikit-learn developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#
# Copyright 2022 Autarky.ai LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""General-purpose tool classes.

These classes provide downstream functionality by exposing methods for
use in their client subclasses.

Classes
-------
PickleMixin()
    Save/load class instances using :mod:`pickle`.
ReprMixin()
    Implement a custom string representation of class instances.

"""

__all__ = ['PickleMixin', 'ReprMixin']

import inspect as _inspect
import os as _os
import pickle as _pickle
import re as _re
import sklearn as _sklearn


class PickleMixin:
    """Mix-in class to save/load class instances as pickled data."""

    def _save(self, *, filename):
        with open(file=filename, mode='wb') as f:
            _pickle.dump(obj=self, file=f, protocol=_pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, *, filename):
        """Load pickled object.

        Parameters
        ----------
        filename : str
            The pickle file to be loaded.
        """
        with open(file=filename, mode='rb') as f:
            return _pickle.load(file=f)

    def save(self, *, filename='Untitled.pickle'):
        """Save object as a pickle.

        Prompts the user for a 'yes'/'no' answer to avoid accidental
        file overwriting.

        Parameters
        ----------
        filename : str, default='Untitled.pickle'
            The file name under which to save the object.
        """
        if _os.path.exists(path=filename):
            prompt = (f"WARNING: '{filename}' already exists. Are you sure "
                      f"you want to overwrite it ([y]/n)? ")
            user_input = input(prompt=prompt)
            if user_input.lower() in ['y', 'yes', '']:
                self._save(filename=filename)
            else:
                pass
        else:
            self._save(filename=filename)


class ReprMixin:
    """Mix-in class enabling a custom string representation of class
    instance objects.
    """

    def __repr__(self, *, N_CHAR_MAX=700):
        # N_CHAR_MAX is the (approximate) maximum number of non-blank
        # characters to render. We pass it as an optional parameter to
        # ease the tests.
        from autarkydotai.utils._pprint import _PrettyPrinter

        N_MAX_ELEMENTS_TO_SHOW = 30  # No. of elements to show in sequences

        # Use ellipses for sequences with a lot of elements. Use brute-
        # force ellipses when there are a lot of non-blank characters.
        pp = _PrettyPrinter(compact=True,
                            n_max_elements_to_show=N_MAX_ELEMENTS_TO_SHOW)
        repr_ = pp.pformat(object=self)
        n_nonblank = len(''.join(repr_.split()))
        if n_nonblank > N_CHAR_MAX:
            # Approx number of characters to keep on both ends.
            lim = N_CHAR_MAX // 2
            regex = r'^(\s*\S){%d}' % lim
            # The regex '^(\s*\S){%d}' % n matches from the start of
            # the string until the nth non-blank character:
            # - ^ matches the start of string
            # - (pattern){n} matches n repetitions of pattern
            # - \s*\S matches a non-blank char following zero or more
            #   blanks.
            left_lim = _re.match(pattern=regex, string=repr_).end()
            right_lim = _re.match(pattern=regex, string=repr_[::-1]).end()
            if '\n' in repr_[left_lim:-right_lim]:
                # The left side and right side aren't on the same line.
                # To avoid weird cuts (e.g. 'categoric...ore'), we need
                # to start the right side with an appropriate newline
                # character so that it renders properly as:
                # categoric...
                # handle_unknown='ignore'.
                # So we add [^\n]*\n, which matches until the next \n.
                regex += r'[^\n]*\n'
                right_lim = _re.match(pattern=regex, string=repr_[::-1]).end()
            if left_lim + len('...') < len(repr_) - right_lim:
                # Only add ellipses if they result in a shorter repr.
                repr_ = repr_[:left_lim] + '...' + repr_[-right_lim:]

        return repr_

    @classmethod
    def _get_param_names(cls):
        """Get the constructor's argument names."""
        init_signature = cls._get_signature()
        if init_signature is None:
            return []
        # Consider the constructor parameters excluding 'self'.
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self']
        # Extract and sort argument names.
        return sorted([p.name for p in parameters])

    @classmethod
    def _get_signature(cls):
        """Get the signature of the object's constructor."""
        # Fetch the constructor.
        init = cls.__init__
        if init is object.__init__:
            # Class has no explicit constructor.
            return None
        return _inspect.signature(obj=init)

    def get_params(self, *, deep=True):
        """Get constructor arguments for this object.

        Parameters
        ----------
        deep : bool, default=True
            If True, return the constructor arguments for this object
            and any applicable nested objects.

        Returns
        -------
        dict
            Constructor argument names mapped to their values.
        """
        return _sklearn.base.BaseEstimator.get_params(self, deep=deep)
